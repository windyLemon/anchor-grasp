from torch.utils.data import Dataset
import glob
from PIL import Image
from torchvision import transforms
import numpy as np
from utils.visual import plot_rectangle
from utils.convert import target2src, decode, src2target, rotation
from utils.generator import GenerateAnchor
from utils.tools import evaluation
import torch


def transforms_img():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return tf


class GraspData(Dataset):
    def __init__(self, path, train=True, scale=(481, 481), k=6):
        self.path = path
        self.scale = scale
        self.y_scale = scale[0] / 480.
        self.x_scale = scale[1] / 640.

        self.angles = [5 / 12 * np.pi - np.pi / 6 * i for i in range(k)]
        self.files = []
        self.pos = []
        self.train = train
        self.tf = transforms_img()

        directory = [self.path + '/0' + str(i) for i in range(0, 10)]
        directory.append(self.path + '/10')

        for d in directory:
            self.files.extend(glob.glob(d + '/*png'))
            self.pos.extend(glob.glob(d + '/*pos*'))

        self.files = sorted(self.files)
        self.pos = sorted(self.pos)

        if train:
            self.files = self.files[0:709]
            self.pos = self.pos[0:709]

        else:
            self.files = self.files[709:]
            self.pos = self.pos[709:]

    def __getitem__(self, item):

        file = self.files[item]
        img = Image.open(file)
        img = img.resize(self.scale)
        positive = []

        with open(self.pos[item]) as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                coordinates = [lines[i].split(), lines[i + 1].split(),
                               lines[i + 2].split(), lines[i + 3].split()]
                positive.append(coordinates)

        positive = self.convert_position(positive)
        positive = rotation(positive, (241, 241), np.array(self.angles))

        select = int(np.random.choice(len(self.angles), 1))
        # select = 1
        positive = positive[select, :, :]

        img = img.rotate(-self.angles[select] * 180 / np.pi)
        img = self.tf(img)

        # print(positive)
        return img, positive

    def __len__(self):
        return len(self.files)

    def convert_position(self, positive):
        pos = []
        for position in positive:
            x0, y0 = position[0]
            x1, y1 = position[1]
            x2, y2 = position[2]
            x3, y3 = position[3]

            pos.append([float(y0) * self.y_scale, float(x0) * self.x_scale,
                        float(y1) * self.y_scale, float(x1) * self.x_scale,
                        float(y2) * self.y_scale, float(x2) * self.x_scale,
                        float(y3) * self.y_scale, float(x3) * self.x_scale])
        return np.array(pos)

# # # [ 400 1353 1519 1899 1497 1334 1481]
# g = GraspData('../CornellData/data', train=True)
# Generate = GenerateAnchor(k=3, scales=[600, 1200., 2500.], ratios=[1., 2.])
# image, pos = g[0]
# image = ((image.numpy().transpose(1, 2, 0) * 0.5) + 0.5)
# anchors = Generate((15, 15))
# # indices, indices_position, indices_edge, indices_theta = evaluation(anchors, pos)
# indices = evaluation(anchors, pos)
# # positive_indices = np.argmax(indices_position, axis=0)
# # positive_values = indices_position[positive_indices, np.arange(indices_position.shape[1])]
# positive_indices = np.argmax(indices, axis=1)
#
# positive_value = indices[np.arange(positive_indices.shape[0]), positive_indices]
# # positive_value_position = indices_position[np.arange(positive_indices.shape[0]), positive_indices]
# # positive_value_edge = indices_edge[np.arange(positive_indices.shape[0]), positive_indices]
# # positive_value_theta = indices_theta[np.arange(positive_indices.shape[0]), positive_indices]
#
# max_eva = np.max(indices, axis=0)
# # print(max_eva)
# pos_anchor_indices = np.where(positive_value > 0.3)[0]
# positive_indices = positive_indices[pos_anchor_indices]
#
# neg_anchor_indices = np.where(positive_value < 0.3)[0]
# print(positive_value[pos_anchor_indices])
# # print(positive_value_position[pos_anchor_indices])
# # print(positive_value_edge[pos_anchor_indices])
# # print(positive_value_theta[pos_anchor_indices])
#
# print(positive_indices)
#
# offset = src2target(pos[positive_indices, :], anchors[pos_anchor_indices, :])
# pos = target2src(offset, anchors[pos_anchor_indices, :])
# print(offset)
#
# pos = decode(pos)
# anchors = decode(anchors)
# # print(positive_values)
# # plot_rectangle(image, pos)
# plot_rectangle(image, anchors[pos_anchor_indices], pos)
