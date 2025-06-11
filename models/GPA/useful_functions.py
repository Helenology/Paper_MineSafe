import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# optimal bandwidth
def compute_optimal_bandwidths(N, alpha):
    bandwidth = np.pi ** (-2 / 7) * (0.04158134 ** (0.5)) ** (5 / 7) * N ** (-(1 + alpha) / 7)
    print("Optimal bandwidth from Rule of Thumb:", bandwidth)
    bandwidth_star = bandwidth ** 2 * 5
    print("Optimal bandwidth* from GPA estimation:", bandwidth_star)
    return bandwidth, bandwidth_star


# def plot_sub_img(test_img, boxA, boxB=None, IsShow=True):
#     show_img_one_channel(test_img, False)
# #     plt.imshow(test_img)
#     if boxA is not None and boxA[0] is not None:
#         x1, y1, x2, y2 = boxA
#         col1 = 'red'
#         plt.vlines(y1, ymin=x1, ymax=x2, color=col1)
#         plt.vlines(y2, ymin=x1, ymax=x2, color=col1)
#         plt.hlines(x1, xmin=y1, xmax=y2, color=col1)
#         plt.hlines(x2, xmin=y1, xmax=y2, color=col1)
#     if boxB is not None:
#         x1, y1, x2, y2 = boxB
#         col = 'yellow'
#         plt.vlines(y1, ymin=x1, ymax=x2, color=col)
#         plt.vlines(y2, ymin=x1, ymax=x2, color=col)
#         plt.hlines(x1, xmin=y1, xmax=y2, color=col)
#         plt.hlines(x2, xmin=y1, xmax=y2, color=col)
#     if IsShow:
#         plt.show()
#     return


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


def compute_classical_density_experiment(p, q, bandwidth, img_index, filepath_list, test_img):
    classic_density_tensor = tf.zeros((p, q), dtype=tf.float32)
    print(f"--Compute the classic density estimations--")
    N0 = len(img_index)

    for i in range(N0):
        # load simulation image
        image_path = filepath_list[img_index[i]]
        simulate_img = load_and_preprocess_image(image_path, (p, q))

        # compute classic nonparametric density estimator
        tmp_tensor = 1 / tf.sqrt(2 * np.pi) * tf.exp(-(simulate_img - test_img) ** 2 / (2 * bandwidth ** 2))
        tmp_tensor = tmp_tensor / (N0 * bandwidth)
        classic_density_tensor += tmp_tensor

    print(f"Successfully compute classical density with N0={N0}")
    return classic_density_tensor