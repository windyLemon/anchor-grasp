import torch
import torch.nn as nn
from torchvision import models
from utils.generator import GenerateAnchor, SampleAnchors
import torch.nn.functional as F
from utils.visual import plot_feature


class ExtractFeature(nn.Module):
    def __init__(self, k=3, scales=[600, 1200., 2500.], ratios=[1., 2.]):
        super(ExtractFeature, self).__init__()
        self.dense = models.densenet121(pretrained=False).features
        self.generate_anchors = GenerateAnchor(k, scales, ratios)
        self.n_anchors = k * len(scales) * len(ratios)
        # self.feature = nn.Sequential(*list(self.res_net.children())[:-1])

    def forward(self, x):
        x = self.dense(x)
        N, C, H, W = x.size()
        anchors = self.generate_anchors((H, W))
        return x, anchors


class RPN(nn.Module):
    def __init__(self, n_anchors):
        super(RPN, self).__init__()

        self.hidden = nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=(1, 1))
        self.classifier = nn.Conv2d(512, n_anchors * 2, kernel_size=(1, 1))
        self.regression = nn.Conv2d(512, n_anchors * 5, kernel_size=(1, 1))

    def forward(self, x):
        # image = x

        x = self.hidden(x)
        x = F.relu(x)
        classifier = self.classifier(x)
        regression = self.regression(x)

        # feature = classifier[0]
        # plot_feature(classifier, image)
        # plt.imshow(feature.permute(1, 2, 0).detach().numpy()[:, :, 0])
        # plt.show()

        regression = regression.permute((0, 2, 3, 1)).contiguous().view(1, -1, 5)

        classifier = classifier.permute((0, 2, 3, 1)).contiguous().view(1, -1, 2)

        # rois = self.sample(classifier, regression, anchors)

        return classifier, regression


class Detector(nn.Module):
    def __init__(self, size=(7, 7)):
        super(Detector, self).__init__()
        self.extract = ExtractFeature()

        self.rpn1 = RPN(self.extract.n_anchors)
        self.rpn2 = RPN(self.extract.n_anchors)
        self.rpn3 = RPN(self.extract.n_anchors)



#
# class GraspModel(nn.Module):
#     def __init__(self, k=3, scales=3, ratios=2):
#         super(GraspModel, self).__init__()
#         self.dense = models.densenet121(pretrained=False).features
#         # print(self.dense)
#         self.generator = GenerateAnchor()
#         self.sample = SampleAnchors()
#         self.conv1 = nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=(1, 1))
#
#         self.classifier = nn.Conv2d(512, k * scales * ratios * 2, kernel_size=(1, 1))
#         self.regression = nn.Conv2d(512, k * scales * ratios * 5, kernel_size=(1, 1))
#
#     def forward(self, x):
#         # image = x
#
#         x = self.dense(x)
#         N, C, H, W = x.size()
#         anchors = self.generator((H, W))
#
#         x = self.conv1(x)
#         x = F.relu(x)
#         classifier = self.classifier(x)
#         regression = self.regression(x)
#
#         # feature = classifier[0]
#         # plot_feature(classifier, image)
#         # plt.imshow(feature.permute(1, 2, 0).detach().numpy()[:, :, 0])
#         # plt.show()
#
#         regression = regression.permute((0, 2, 3, 1)).contiguous().view(N, -1, 5)
#
#         classifier = classifier.permute((0, 2, 3, 1)).contiguous().view(N, -1, 2)
#
#         rois = self.sample(classifier, regression, anchors)
#
#         return classifier, regression, anchors, rois

