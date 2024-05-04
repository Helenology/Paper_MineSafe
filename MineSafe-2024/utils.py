import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re


def load_img_np(path):
    img = Image.open(path)
    img = np.array(img).astype(np.int32) / 255.0
    img = img.mean(axis=2)
    return img



def show_img_one_channel(img, IsShow=False):
#     print(f"Image shape:{img.shape}")
    img = np.expand_dims(img, axis=-1)
    new_img = np.ones((img.shape[0], img.shape[1], 3)) # 生成一个空的数组，大小和原来的数组相同
    new_img *= img # 将每个channel赋值为同样的数组
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