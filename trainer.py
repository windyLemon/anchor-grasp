import torch
from torch.utils.data import DataLoader
from utils.generator import ProposalAnchor
from model.net import Detector
from data.cornell import GraspData
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms
from PIL import Image
from utils.tools import numpy2tensor, nms, evaluation
from utils.convert import target2src, decode
from utils.visual import plot_rectangle


class Trainer:
    def __init__(self, model, data, path, cascade=3):
        self.model = model
        self.data_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=1)
        self.proposal = ProposalAnchor()
        self.optim = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=1e-3)
        self.path = path
        # self.writer = SummaryWriter('runs/log')
        self.total_loss = 0
        self.total_cla_loss = 0
        self.total_reg_loss = 0

        # self.cascade_total_loss = {0: self.total_loss1, 1: self.total_loss2, 2: self.total_loss3}
        # self.cascade_total_cla_loss = {0: self.total_cla_loss1, 1: self.total_cla_loss2, 2: self.total_cla_loss3}
        # self.cascade_total_reg_loss = {0: self.total_reg_loss1, 1: self.total_reg_loss2, 2: self.total_reg_loss3}
        self.cascade_total_loss = []
        self.cascade_total_cla_loss = []
        self.cascade_total_reg_loss = []
        self.cascade = cascade

    def train(self):
        self.model.train()

        cascade_rpn = {0: self.model.rpn1, 1: self.model.rpn2, 2: self.model.rpn3}
        cascade_up_threshold = {0: 0.4, 1: 0.5, 2: 0.6}
        cascade_down_threshold = {0: 0.2, 1: 0.2, 2: 0.3}

        if os.path.exists(self.path):
            checkpoint = self.load(self.path)
            self.model.load_state_dict(checkpoint['model'])
            epoch = checkpoint['epoch']
            i = checkpoint['i']
            print('************load model***************')
        else:
            epoch = 0
            i = 0

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print('***********cuda is avaiable**********')

        while True:
            for data in self.data_loader:

                image, positive = data
                image = numpy2tensor(image)

                feature, anchors = self.model.extract(image)
                self.cascade_total_loss.clear()
                self.cascade_total_reg_loss.clear()
                self.cascade_total_cla_loss.clear()

                for c in range(self.cascade):

                    print(anchors.shape)
                    classifier, regression = cascade_rpn[c](feature)
                    label, offset, pos_anchor_indices = self.proposal(positive, anchors, cascade_up_threshold[c],
                                                                      cascade_down_threshold[c])

                    classifier = classifier[0]
                    regression = regression[0]

                    label = torch.from_numpy(label).long()
                    offset = torch.from_numpy(offset)
                    label = numpy2tensor(label)
                    offset = numpy2tensor(offset)
                    offset = offset.float()

                    anchors = self.integrate(classifier, regression, anchors)

                    regression = regression[pos_anchor_indices]
                    cla_loss, reg_loss, loss = self.loss(classifier, regression, label, offset)
                    self.cascade_total_cla_loss.append(cla_loss)
                    self.cascade_total_reg_loss.append(reg_loss)
                    self.cascade_total_loss.append(loss)

                total_loss = sum(self.cascade_total_loss)
                cla_loss = sum(self.cascade_total_cla_loss)
                reg_loss = sum(self.cascade_total_reg_loss)

                if torch.isnan(total_loss):
                    continue

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                self.save(epoch, i, model, self.path)

                self.total_loss = self.total_loss + total_loss
                self.total_cla_loss = self.total_cla_loss + cla_loss
                self.total_reg_loss = self.total_reg_loss + reg_loss

                print('epoch is {}, i is {}, and loss is {}, cla_loss is {}, reg_loss is{}'
                      .format(epoch, i, total_loss, cla_loss, reg_loss))

                i = i + 1

            i = 0
            epoch = epoch + 1

    def integrate(self, classifier, regression, anchor):
        classifier = F.softmax(classifier, dim=1)

        classifier = classifier.detach().numpy()
        regression = regression.detach().numpy()

        right_indices = np.argsort(classifier[:, 1])[::-1]
        num = np.where(classifier[:, 1] > 0.5)[0].shape[0]

        print(num)
        right_indices = right_indices[:num]

        right_anchor = anchor[right_indices]
        right_regression = regression[right_indices]

        right = target2src(right_regression, right_anchor)
        anchor[right_indices] = right

        return anchor

    def loss(self, classifier, regression, label, offset):
        cla_loss = F.cross_entropy(classifier, label, ignore_index=-1)

        # smooth L1 loss
        diff = (regression - offset).abs()
        # print(diff.size())

        condition = (diff < 1).float()
        # diff = diff[diff < 1] - 0.5
        reg_loss = condition * (diff ** 2) + (1 - condition) * (diff - 0.5)
        reg_loss = torch.sum(reg_loss) / diff.size()[0]
        loss = cla_loss + reg_loss

        return cla_loss, reg_loss, loss

    def save(self, epoch, i, model, path):
        save_dic = dict()
        if i % 100 == 0 and i != 0:
            if torch.cuda.is_available():
                model = model.cpu()
            save_dic['model'] = model.state_dict()
            save_dic['epoch'] = epoch
            save_dic['i'] = i
            torch.save(save_dic, path)

            self.writer.add_scalar('training loss', self.total_loss / 100., epoch * len(self.data_loader) + i)
            self.writer.add_scalar('classifier loss', self.total_cla_loss / 100., epoch * len(self.data_loader) + i)
            self.writer.add_scalar('regression loss', self.total_reg_loss / 100., epoch * len(self.data_loader) + i)
            self.total_loss = 0
            self.total_cla_loss = 0
            self.total_reg_loss = 0

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        return checkpoint

    def test(self, image, pos):

        self.model.eval()
        if os.path.exists(self.path):
            checkpoint = self.load(self.path)
            self.model.load_state_dict(checkpoint['model'])
            print('************load model***************')

        self.model.eval()
        image = torch.unsqueeze(image, dim=0)

        feature, anchors = self.model.extract(image)

        classifier, regression = self.model.rpn1(feature)
        classifier = classifier[0]
        regression = regression[0]
        anchors = self.integrate(classifier, regression, anchors)

        classifier, regression = self.model.rpn2(feature)
        classifier = classifier[0]
        regression = regression[0]
        anchors = self.integrate(classifier, regression, anchors)
        # #
        classifier, regression = self.model.rpn3(feature)
        classifier = classifier[0]
        regression = regression[0]
        anchors = self.integrate(classifier, regression, anchors)

        classifier = F.softmax(classifier, dim=1)

        classifier = classifier.detach().numpy()
        regression = regression.detach().numpy()

        right_indices = np.argsort(classifier[:, 1])[::-1]
        num = np.where(classifier[:, 1] > 0.9)[0].shape[0]
        print(num)
        right_indices = right_indices[:num]

        # print(right_indices)

        # right_indices_cla = [2693, 2709, 2715, 3026, 3027, 3040]
        right_anchor = anchors[right_indices]
        right_regression = regression[right_indices]

        classifier = classifier[right_indices]

        right = right_anchor
        # print(right_regression)
        # right = target2src(right_regression, right_anchor)

        keep = nms(right, 0.5)
        right = right[keep, :]

        eva = evaluation(right, pos)
        eva_indices = np.argmax(eva, axis=1)
        eva = eva[np.arange(eva.shape[0]), eva_indices]

        # print(classifier[right_indices_cla, 1])
        # print(classifier[right_indices[0:10], 1])
        print(eva)
        # print(keep)
        # print(classifier[311, :])
        return right


def load(path):
    image = Image.open(path)
    image = image.resize((481, 481))
    image = image.rotate(0)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    image = tf(image)

    return image


if __name__ == '__main__':
    g = GraspData('CornellData/data', train=True)
    image, pos = g[20]
    #
    model = Detector()
    data = GraspData(path='CornellData/data')
    t = Trainer(model, data, path='para.tar')
    # t.train()
    # print(all_pos)

    # image = load('test1.jpeg')
    #
    right = t.test(image, pos)
    right = decode(right)
    pos = decode(pos)
    image = ((image.numpy().transpose(1, 2, 0) * 0.5) + 0.5)
    # plot_rectangle(image, pos)
    # print(right)
    plot_rectangle(image, right, pos)
    # visual(image, right)

    # [2693 2699 2705 2708 2709 2714 2715 3028 3029 3040 3041]