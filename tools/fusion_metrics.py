
import numpy as np

def entropy_from_counts(h):
    p = h / h.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def entropy(img):                      # img: uint8, H×W
    return entropy_from_counts(np.bincount(img.ravel(), minlength=256))

def std_dev(img):
    x = img.astype(np.float64)
    return np.sqrt(((x - x.mean()) ** 2).mean())

def spatial_frequency(img):
    x = img.astype(np.float64)
    rf = np.sqrt(((x[:, 1:] - x[:, :-1]) ** 2).mean())
    cf = np.sqrt(((x[1:, :] - x[:-1, :]) ** 2).mean())
    return np.sqrt(rf ** 2 + cf ** 2)

def joint_entropy(a_img, f_img):
    a = a_img.ravel().astype(np.int64)
    f = f_img.ravel().astype(np.int64)
    joint = np.bincount(a * 256 + f, minlength=256 * 256).reshape(256, 256)
    return entropy_from_counts(joint)

def mutual_information(a_img, f_img):
    return entropy(a_img) + entropy(f_img) - joint_entropy(a_img, f_img)

def mi_fusion(a_img, b_img, f_img):
    return mutual_information(a_img, f_img) + mutual_information(b_img, f_img)
