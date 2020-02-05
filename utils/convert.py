import numpy as np


def rotation(position, center, angles):
    angles = angles[:, np.newaxis]
    image_y, image_x = center

    y0, x0 = position[:, 0] - image_y, position[:, 1] - image_x
    y1, x1 = position[:, 2] - image_y, position[:, 3] - image_x
    y2, x2 = position[:, 4] - image_y, position[:, 5] - image_x
    y3, x3 = position[:, 6] - image_y, position[:, 7] - image_x

    y0, x0 = image_y + y0 * np.cos(angles) + x0 * np.sin(angles), image_x + x0 * np.cos(angles) - y0 * np.sin(angles)
    y1, x1 = image_y + y1 * np.cos(angles) + x1 * np.sin(angles), image_x + x1 * np.cos(angles) - y1 * np.sin(angles)
    y2, x2 = image_y + y2 * np.cos(angles) + x2 * np.sin(angles), image_x + x2 * np.cos(angles) - y2 * np.sin(angles)
    y3, x3 = image_y + y3 * np.cos(angles) + x3 * np.sin(angles), image_x + x3 * np.cos(angles) - y3 * np.sin(angles)

    center_y, center_x = (y0 + y2) / 2., (x0 + x2) / 2.
    theta = (y1 - y0) / (x1 - x0 + 0.001)
    theta = -np.arctan(theta)
    along_grasp = np.sqrt((y0 - y1) ** 2 + (x0 - x1) ** 2)
    vertical_grasp = np.sqrt((y0 - y3) ** 2 + (x0 - x3) ** 2)

    center_y = center_y[:, :, np.newaxis]
    center_x = center_x[:, :, np.newaxis]
    theta = theta[:, :, np.newaxis]
    along_grasp = along_grasp[:, :, np.newaxis]
    vertical_grasp = vertical_grasp[:, :, np.newaxis]

    return np.concatenate((center_y, center_x, along_grasp, vertical_grasp, theta), axis=2)


def encode(position):
    '''
    convert
    :param position: math (k, 8)
    :return: target (k, 5), center_x, center_y, h, w, theta
    '''

    y0, x0 = position[:, 0], position[:, 1]
    y1, x1 = position[:, 2], position[:, 3]
    y2, x2 = position[:, 4], position[:, 5]
    y3, x3 = position[:, 6], position[:, 7]

    center_y, center_x = (y0 + y2) / 2., (x0 + x2) / 2.
    theta = (y1 - y0) / (x1 - x0 + 0.1)
    theta = -np.arctan(theta)
    along_grasp = np.sqrt((y0 - y1) ** 2 + (x0 - x1) ** 2)
    vertical_grasp = np.sqrt((y0 - y3) ** 2 + (x0 - x3) ** 2)

    center_y = center_y.reshape(-1, 1)
    center_x = center_x.reshape(-1, 1)
    theta = theta.reshape(-1, 1)
    along_grasp = along_grasp.reshape(-1, 1)
    vertical_grasp = vertical_grasp.reshape(-1, 1)

    return np.concatenate((center_y, center_x, along_grasp, vertical_grasp, theta), axis=1)


def decode(position):
    '''
    resume
    :param position: (k, 5)
    :return:position (k, 8)
    '''

    theta = position[:, 4]
    rectangle_theta = np.arctan(position[:, 3] / position[:, 2])

    offset_top = theta + rectangle_theta
    offset_down = theta - rectangle_theta

    diagonal = np.sqrt(position[:, 3] ** 2 + position[:, 2] ** 2) / 2.
    offset_topx = diagonal * np.cos(offset_top)
    offset_topy = diagonal * np.sin(offset_top)

    offset_downx = diagonal * np.cos(offset_down)
    offset_downy = diagonal * np.sin(offset_down)

    right_top_x = (position[:, 1] + offset_topx).reshape(-1, 1)
    right_top_y = (position[:, 0] - offset_topy).reshape(-1, 1)
    right_down_x = (position[:, 1] + offset_downx).reshape(-1, 1)
    right_down_y = (position[:, 0] - offset_downy).reshape(-1, 1)

    left_top_x = (position[:, 1] - offset_topx).reshape(-1, 1)
    left_top_y = (position[:, 0] + offset_topy).reshape(-1, 1)
    left_down_x = (position[:, 1] - offset_downx).reshape(-1, 1)
    left_down_y = (position[:, 0] + offset_downy).reshape(-1, 1)

    return np.concatenate((right_top_y, right_top_x,
                           left_down_y, left_down_x,
                           left_top_y, left_top_x,
                           right_down_y, right_down_x), axis=1)


def src2target(src, anchor):
    '''
    src to target
    :param anchor: math (k, 5), selected anchors
    :param src: math (k, 5)
    :return: target (k, 5)
    '''

    distance = np.sqrt(anchor[:, 2] ** 2 + anchor[:, 3] ** 2)
    t_y = ((src[:, 0] - anchor[:, 0]) / distance).reshape(-1, 1)
    t_x = ((src[:, 1] - anchor[:, 1]) / distance).reshape(-1, 1)

    t_along = np.log(src[:, 2] / anchor[:, 2]).reshape(-1, 1)
    t_vertical = np.log(src[:, 3] / anchor[:, 3]).reshape(-1, 1)

    t_theta = (180 * (src[:, 4] - anchor[:, 4]) / np.pi / 60).reshape(-1, 1)

    return np.concatenate((t_y, t_x, t_along, t_vertical, t_theta), axis=1)


def target2src(target, anchor):
    distance = np.sqrt(anchor[:, 2] ** 2 + anchor[:, 3] ** 2)
    center_y = (target[:, 0] * distance + anchor[:, 0]).reshape(-1, 1)
    center_x = (target[:, 1] * distance + anchor[:, 1]).reshape(-1, 1)

    along_grasp = (np.exp(target[:, 2]) * anchor[:, 2]).reshape(-1, 1)
    vertical_grasp = (np.exp(target[:, 3]) * anchor[:, 3]).reshape(-1, 1)

    theta = (anchor[:, 4] + 60 * np.pi * target[:, 4] / 180).reshape(-1, 1)

    return np.concatenate((center_y, center_x, along_grasp, vertical_grasp, theta), axis=1)


def translation_rotation(anchor, bbox):
    '''
    convert anchor to bbox coordinate
    :param anchor: math (S, 5)
    :param bbox: math (K, 5)
    :return: math (S, K, 2), each element represent anchor[i] and bbox[j] center convert
    '''

    rotation_theta = bbox[:, 4]
    # print(rotation_theta)

    sin_theta = np.sin(rotation_theta)
    cos_theta = np.cos(rotation_theta)

    translation = anchor[:, np.newaxis, 0:2] - bbox[:, 0:2]
    # print(translation[1899])

    rotation_x = translation[:, :, 1] * cos_theta - translation[:, :, 0] * sin_theta
    rotation_y = translation[:, :, 1] * sin_theta + translation[:, :, 0] * cos_theta

    # print(rotation_x[1899], rotation_y[1899])

    return rotation_x, rotation_y
