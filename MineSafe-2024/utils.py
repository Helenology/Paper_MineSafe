import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re


def load_img_np(path):
    img = Image.open(path)
    img = np.array(img).astype(np.int32) / 255.0
    img = img.mean(axis=2)
    return img


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