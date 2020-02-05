import matplotlib.pyplot as plt
import numpy as np


def visual(img, pos_theta):
    plt.imshow(img)
    for position in pos_theta:
        dx = 1.
        dy = 1. * np.tan(position[2])

        dx = dx / np.sqrt(dx ** 2 + dy ** 2)
        dy = dy / np.sqrt(dx ** 2 + dy ** 2)
        plt.arrow(position[1], position[0], dx * 20, -dy * 20,
                  length_includes_head=True, head_width=5, head_length=10, fc='r', ec='b')
    plt.show()


def plot_rectangle(img, coordinates, pos):
    plt.figure(0)
    plt.imshow(img)
    for coordinate in coordinates:
        plt.plot((coordinate[1], coordinate[3]), (coordinate[0], coordinate[2]), linewidth=2, color='r')
        plt.plot((coordinate[3], coordinate[5]), (coordinate[2], coordinate[4]), linewidth=2, color='g')
        plt.plot((coordinate[5], coordinate[7]), (coordinate[4], coordinate[6]), linewidth=2, color='r')
        plt.plot((coordinate[7], coordinate[1]), (coordinate[6], coordinate[0]), linewidth=2, color='g')
    plt.figure(1)
    plt.imshow(img)
    for coordinate in pos:
        plt.plot((coordinate[1], coordinate[3]), (coordinate[0], coordinate[2]), linewidth=2, color='r')
        plt.plot((coordinate[3], coordinate[5]), (coordinate[2], coordinate[4]), linewidth=2, color='g')
        plt.plot((coordinate[5], coordinate[7]), (coordinate[4], coordinate[6]), linewidth=2, color='r')
        plt.plot((coordinate[7], coordinate[1]), (coordinate[6], coordinate[0]), linewidth=2, color='g')
    plt.show()


def plot_feature(feature, image):
    feature = feature[0]
    image = image[0]
    feature = feature.detach().numpy()
    image = image.detach().numpy()

    # print(feature.shape)

    feature = feature.transpose(1, 2, 0)
    image = image.transpose(1, 2, 0)

    k = 6
    plt.figure(0)
    plt.imshow(image)
    for i in range(1, k + 1):
        plt.figure(i)
        plt.imshow(feature[:, :, i + 1])

    plt.show()
