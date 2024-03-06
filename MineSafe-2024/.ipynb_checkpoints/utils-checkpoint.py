import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_img_np(path):
	img = Image.open(path)
	img = np.array(img).astype(np.int32) / 255.0
	img = img.mean(axis=2)
	return img