import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_location_filter(p, q, h, width=5):
    # a weight matrix
    location_weight = np.zeros([width, width, 2])
    # weight by row
    for i in range(width):
        location_weight[i, 0:width, 0] = abs(i-width//2) / p
    # weight by column
    for j in range(width):
        location_weight[0:width, j, 1] = abs(j-width//2) / q
    # convert to tensor
    location_weight = tf.convert_to_tensor(location_weight, dtype=tf.float32)
    
    # product of row weight and column weight
    location_filter = K_tf(location_weight[:, :, 0], h) * K_tf(location_weight[:, :, 1], h)
    
    # normalize the sum to 1
    location_filter /= tf.reduce_sum(location_filter)
    
    location_filter = tf.reshape(location_filter, [width, width, 1, 1])
    return location_filter


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


# optimal bandwidth
def compute_optimal_bandwidths(N, alpha):
    bandwidth = np.pi**(-2/7) * (0.04158134**(0.5))**(5/7) * N**(-(1+alpha)/7)
    print("Optimal bandwidth from Rule of Thumb:", bandwidth)
    bandwidth_star = bandwidth**2 * 5
    print("Optimal bandwidth* from GPA estimation:", bandwidth_star)
    return bandwidth, bandwidth_star


def plot_sub_img(test_img, boxA, boxB=None, IsShow=True):
    show_img_one_channel(test_img, False)
#     plt.imshow(test_img)
    if boxA is not None and boxA[0] is not None:
        x1, y1, x2, y2 = boxA
        col1 = 'red'
        plt.vlines(y1, ymin=x1, ymax=x2, color=col1)
        plt.vlines(y2, ymin=x1, ymax=x2, color=col1)
        plt.hlines(x1, xmin=y1, xmax=y2, color=col1)
        plt.hlines(x2, xmin=y1, xmax=y2, color=col1)
    if boxB is not None:
        x1, y1, x2, y2 = boxB
        col = 'yellow'
        plt.vlines(y1, ymin=x1, ymax=x2, color=col)
        plt.vlines(y2, ymin=x1, ymax=x2, color=col)
        plt.hlines(x1, xmin=y1, xmax=y2, color=col)
        plt.hlines(x2, xmin=y1, xmax=y2, color=col)
    if IsShow:
        plt.show()
    return


def K_tf(x, h): # a standard normal kernel
    outcome = 1 / tf.math.sqrt(2 * np.pi) * tf.math.exp(-(x)**2 / (2 * h**2))
    return outcome


def load_and_preprocess_image(path, shape):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, dtype=tf.float32)
    image = tf.image.resize(image, shape)
    image = tf.reduce_mean(image, axis=2)
    return image


def load_and_preprocess_image_3_channels(path, shape):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, dtype=tf.float32)
    image = tf.image.resize(image, shape)
    return image


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


def produce_coordinate(test_data, i):
    y1, x1, y2, x2 = test_data.loc[i, ["Left_Y", "Left_X", "Right_Y", "Right_X"]]
    try:
        return int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
    except:
        return x1, y1, x2, y2

    
def compute_classical_density_experiment(p, q, bandwidth, img_index, filepath_list, test_img):
    classic_density_tensor = tf.zeros((p, q), dtype=tf.float32)
    print(f"--Compute the classic density estimations--")
    N0 = len(img_index)
    
    for i in range(N0):
        # load simulation image
        image_path = filepath_list[img_index[i]]
        simulate_img = load_and_preprocess_image(image_path, (p, q))
        
        # compute classic nonparametric density estimator
        tmp_tensor = 1 / tf.sqrt(2*np.pi) * tf.exp(-(simulate_img - test_img)**2 / (2 * bandwidth**2))
        tmp_tensor = tmp_tensor / (N0 * bandwidth)
        classic_density_tensor += tmp_tensor
        
    print(f"Successfully compute classical density with N0={N0}")
    return classic_density_tensor