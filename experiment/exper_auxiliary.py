#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/6 11:14
# @Author  : Helenology
# @Site    : 
# @File    : exper_auxiliary.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


def plot_sub_img(test_img, boxA, boxB=None, colA='red', colB='yellow'):
    # show_img_one_channel(test_img, False)
    plt.imshow(test_img)
    if boxA is not None and boxA[0] is not None:
        x1, y1, x2, y2 = boxA
        plt.vlines(y1, ymin=x1, ymax=x2, color=colA)
        plt.vlines(y2, ymin=x1, ymax=x2, color=colA)
        plt.hlines(x1, xmin=y1, xmax=y2, color=colA)
        plt.hlines(x2, xmin=y1, xmax=y2, color=colA)
    if boxB is not None:
        x1, y1, x2, y2 = boxB
        plt.vlines(y1, ymin=x1, ymax=x2, color=colB)
        plt.vlines(y2, ymin=x1, ymax=x2, color=colB)
        plt.hlines(x1, xmin=y1, xmax=y2, color=colB)
        plt.hlines(x2, xmin=y1, xmax=y2, color=colB)
    plt.axis('off')
    plt.show()


def compute_IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def rescale_yolo_bounding_boxes(raw_shape, yolo_shape, yolo_box):
    """
    Rescales bounding box coords according to new image size
    :param original_shape: (W, H)
    :param new_shape: (W, H)
    :param bounding_boxes: [[x1, y1, x2, y2], ...]
    :return: scaled bbox coords
    """
    scale = raw_shape[1] / yolo_shape[1]
    yolo_box = np.array(yolo_box, dtype=np.float64)
    yolo_box *= scale
    yolo_box = np.clip(yolo_box, a_min=0, a_max=None)
    yolo_box = yolo_box.astype(np.uint32)
    return yolo_box


def obtain_yolo_box_class(y_pred, raw_shape, yolo_shape=(640, 640)):
    # classes
    predicted_classes = y_pred['classes'][0][y_pred['classes'][0] >= 0]

    # pick out the boxes
    predicted_boxes = []
    for item in y_pred['boxes'][0]:
        if item[0] != -1:
            predicted_boxes.append(list(item))
    # rescale back the bounding box to the raw image size
    predicted_boxes = rescale_yolo_bounding_boxes(raw_shape, yolo_shape, predicted_boxes)
    # adjust the x, y of the bounding box
    boxes = np.zeros_like(predicted_boxes)
    boxes[:, 0] = predicted_boxes[:, 1]
    boxes[:, 1] = predicted_boxes[:, 0]
    boxes[:, 2] = predicted_boxes[:, 3]
    boxes[:, 3] = predicted_boxes[:, 2]
    if boxes.shape[0] > 2:
        boxes = boxes[0:2, ]
        predicted_classes = predicted_classes[0:2, ]
    return boxes, predicted_classes


def load_and_preprocess_image_3_channels(path, shape):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, dtype=tf.float32)
    image = tf.image.resize(image, shape)
    return image


def produce_coordinate(test_data, i):
    y1, x1, y2, x2 = test_data.loc[i, ["Left_Y", "Left_X", "Right_Y", "Right_X"]]
    try:
        return int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
    except:
        return x1, y1, x2, y2


def load_and_preprocess_image(path, shape):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, dtype=tf.float32)
    image = tf.image.resize(image, shape)
    image = tf.reduce_mean(image, axis=2)
    return image
