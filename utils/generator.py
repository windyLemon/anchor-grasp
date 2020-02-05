import numpy as np
import torch
from utils.tools import evaluation
from utils.convert import src2target, target2src
import torch.nn.functional as F


class ProposalAnchor:
    def __init__(self, n_sample=128, ratio=0.25):
        self.n_sample = n_sample
        self.ratio = ratio

    def __call__(self, positive, anchor, up_threshold=0.6, down_threshold=0.3):
        n_positive = int(self.n_sample * self.ratio)
        n_negative = int(self.n_sample - n_positive)

        n = anchor.shape[0]
        self.label = np.zeros(n)
        self.label.fill(-1)

        # positive is a tensor, anchor is a numpy array
        positive = positive.numpy()[0]

        indices_position = evaluation(anchor, positive)
        positive_indices = np.argmax(indices_position, axis=1)
        positive_value = indices_position[np.arange(positive_indices.shape[0]), positive_indices]

        # max_eva = np.max(indices_position, axis=0)
        # print(max_eva)
        pos_anchor_indices = np.where(positive_value > up_threshold)[0]
        positive_indices = positive_indices[pos_anchor_indices]

        neg_anchor_indices = np.where(positive_value < down_threshold)[0]
        # print(pos_anchor_indices)
        # print(neg_anchor_indices)
        # print(positive.shape[0])

        if len(pos_anchor_indices) > 0:
            indices = np.random.choice(len(pos_anchor_indices), n_positive)
            pos_anchor_indices = pos_anchor_indices[indices]
            positive_indices = positive_indices[indices]

        if len(neg_anchor_indices) > 0:
            indices = np.random.choice(len(neg_anchor_indices), n_negative)
            neg_anchor_indices = neg_anchor_indices[indices]
            # negative_indices = negative_indices[indices]

        self.label[neg_anchor_indices] = 0
        self.label[pos_anchor_indices] = 1

        offset = src2target(positive[positive_indices, :], anchor[pos_anchor_indices, :])

        return self.label, offset, pos_anchor_indices

    def grid(self, positive, feature_size=(15, 15)):
        y_length, x_length = feature_size
        y_center, x_center = positive[:, 0], positive[:, 1]

        y_indices, x_indices = y_center / feature_size, x_center / feature_size


class GenerateAnchor:
    def __init__(self, k=3, scales=[600., 1200., 2500.], ratios=[1., 2.]):
        self.k = k
        self.scales = scales
        self.ratios = ratios

    def __call__(self, feature_size, stride=32, base_size=38):
        '''
        use feature map size to generate anchors, ResNet scale is 32, and base_size is 38
        :param feature_size: the ResNet final layer
        :param stride: 32
        :param base_size: 38
        :return:
        '''
        center_x, center_y = base_size / 2, base_size / 2
        H, W = feature_size
        base_anchor = []
        avg_theta = np.pi / (self.k + 1)

        for i in range(self.k):
            theta = np.pi / 2 - avg_theta * (i + 1)
            for scale in self.scales:
                for ratio in self.ratios:
                    h = np.sqrt(scale * ratio)
                    w = np.sqrt(scale / ratio)
                    anchor = np.array([[center_y, center_x, h, w, theta]])
                    base_anchor.append(anchor)
        base_anchor = np.concatenate(base_anchor, axis=0)

        anchors = []
        for i in range(H):
            for j in range(W):
                anchor = base_anchor + np.array([[i, j, 0, 0, 0]]) * stride
                anchors.append(anchor)

        anchors = np.concatenate(anchors, axis=0)
        return anchors


class SampleAnchors:
    def __init__(self, num=512):
        self.num = num

    def __call__(self, scores, offset, anchors):
        scores = F.softmax(scores)
        scores = torch.argsort(scores[0, :, 1])[::-1]

        indices = scores[:self.num]
        anchors = anchors[0, indices, :]
        offset = offset[0, indices, :]

        anchors = target2src(offset, anchors)

        return anchors




