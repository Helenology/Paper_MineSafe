import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import tensorflow as tf


def K_np(x, h):  # a standard normal kernel
    outcome = 1 / np.sqrt(2 * np.pi) * np.exp(-(x)**2 / (2 * h**2))
    return outcome


def K_tf(x, h):  # a standard normal kernel
    outcome = 1 / tf.math.sqrt(2 * np.pi) * tf.math.exp(-(x)**2 / (2 * h**2))
    return outcome


def get_location_filter(p, q, h, width=5):
    # a weight matrix
    location_weight = np.zeros([width, width, 2])
    # weight by row
    for i in range(width):
        location_weight[i, 0:width, 0] = abs(i - width // 2) / p
    # weight by column
    for j in range(width):
        location_weight[0:width, j, 1] = abs(j - width // 2) / q
    # convert to tensor
    location_weight = tf.convert_to_tensor(location_weight, dtype=tf.float32)

    # product of row weight and column weight
    location_filter = K_tf(location_weight[:, :, 0], h) * K_tf(location_weight[:, :, 1], h)

    # normalize the sum to 1
    location_filter /= tf.reduce_sum(location_filter)

    location_filter = tf.reshape(location_filter, [width, width, 1, 1])
    return location_filter


def load_and_preprocess_image(path, shape):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, dtype=tf.float32)
    image = tf.image.resize(image, shape)
    image = tf.reduce_mean(image, axis=2)
    return image


def load_img_np(path):
    img = Image.open(path)
    img = np.array(img).astype(np.int32) / 255.0
    img = img.mean(axis=2)
    return img


def MSE(x, y):
    """Compute the Mean Square Error."""
    mse = tf.reduce_mean((x - y) ** 2)
    return mse


def show_img_one_channel(img, IsShow=False):
    #     print(f"Image shape:{img.shape}")
    img = np.expand_dims(img, axis=-1)
    new_img = np.ones((img.shape[0], img.shape[1], 3))  # 生成一个空的数组，大小和原来的数组相同
    new_img *= img  # 将每个channel赋值为同样的数组
    plt.imshow(new_img)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    if IsShow:
        plt.show()
    return


def show_img_3_channels(img, IsShow=False):
    #     print(f"Image shape:{img.shape}")
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    if IsShow:
        plt.show()
    return


def assert_same_gtraw(name1, name2):
    name1 = re.findall(r"gt.+\.png", name1)[0]
    index1 = re.findall("\d+", name1)[0]
    name2 = re.findall(r"in.+\.jpg", name2.split('/')[-1])[0]
    index2 = re.findall("\d+", name2)[0]
    assert index1 == index2


def compute_metrics(total_fp, total_fn, total_tp):
    Precision = total_tp / (total_tp + total_fp)
    Recall = total_tp / (total_tp + total_fn)
    f1 = 2 * Precision * Recall / (Precision + Recall)
    return f1, Recall, Precision
