import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.convert import translation_rotation


def grasp_iou(bbox, anchor):
    '''
    use projection to calculate grasp iou, like IOU. both h and w choice the line
    across the center point
    :param bbox: math (k, 5) (center_y, center_x, along_grasp, vertical_grasp, theta)
    :param anchor: math (s, 5) (center_y, center_x, along_grasp, vertical_grasp, theta)
    :return: (s, k) element [i, j] is bbox[i] and anchor[j] grasp iou
    '''

    indices_along = 1. / (bbox[:, 2] / anchor[:, 2, np.newaxis] +
                          anchor[:, 2, np.newaxis] / bbox[:, 2])
    indices_vertical = 1. / (bbox[:, 3] / anchor[:, 3, np.newaxis] +
                             anchor[:, 3, np.newaxis] / bbox[:, 3])

    indices_edge = indices_along + indices_vertical

    indices_center = np.sqrt((bbox[:, 0] - anchor[:, 0, np.newaxis]) ** 2 +
                             (bbox[:, 1] - anchor[:, 1, np.newaxis]) ** 2)

    indices_center = indices_center / np.sqrt(bbox[:, 2] * bbox[:, 3])

    indices_theta = np.abs(anchor[:, 4, np.newaxis] - bbox[:, 4])

    return indices_edge, indices_center, indices_theta


def evaluation(anchor, bbox):
    '''
    calculate the evaluation of between anchor
    :param anchor: (S, 5)
    :param bbox: (K, 5)
    :return: (S, K) each element indicates the evaluation of anchor i and bbox j
    '''

    along_max = np.maximum(bbox[:, 2], anchor[:, 2, np.newaxis])
    along_min = np.minimum(bbox[:, 2], anchor[:, 2, np.newaxis])
    vertical_max = np.maximum(bbox[:, 3], anchor[:, 3, np.newaxis])
    vertical_min = np.minimum(bbox[:, 3], anchor[:, 3, np.newaxis])

    indices_along = along_min / along_max
    indices_vertical = vertical_min / vertical_max
    indices_edge = indices_along * indices_vertical

    distance = np.sqrt((bbox[:, 0] - anchor[:, 0, np.newaxis]) ** 2 +
                       (bbox[:, 1] - anchor[:, 1, np.newaxis]) ** 2)
    offset_theta = abs(bbox[:, 4] - anchor[:, 4, np.newaxis])

    indices_position = np.exp(-distance / 32)
    indices_theta = np.cos(offset_theta / 180 * np.pi)

    indices = (indices_edge * indices_position * indices_theta)

    return indices


def nms(anchors, threshold=0.7):
    '''
    select the best anchors from the set
    :param threshold:
    :param anchors: (k, 5)
    :return: (s, 5) and s < k
    '''
    order = np.arange(anchors.shape[0])
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        first = anchors[np.newaxis, i, :]
        others = anchors[order[1:], :]
        eva = evaluation(first, others)
        indices = np.where(eva < threshold)[1]

        order = order[indices + 1]
    return keep


def numpy2tensor(array):
    if torch.cuda.is_available():
        array = array.cuda()
    return array


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets, ignore_index=-1):
        N, C = inputs.size()[0], inputs.size()[1]
        p = F.softmax(inputs)


        return loss