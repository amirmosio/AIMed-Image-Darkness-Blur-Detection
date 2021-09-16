import cv2
import numpy as np

from Utils.resize_img_with_bound import resize


def convert_scale_abs(img):
    img = img.copy()
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def calhe_filter_with_lab(img):
    img = img.copy()

    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def apply_motion_blur(img, kernel_size=30):
    # Specify the kernel size.

    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    # kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    # kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    # kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    # Normalize.
    i = 0
    for j in range(int(kernel_size / 2)):
        kernel_v[j][i] = 1
        kernel_v[j][i + 1] = 1
        i += 2
    # kernel_v[2][6] = 1
    # kernel_v[3][7] = 1
    # kernel_v[3][8] = 1
    # for i in range(kernel_size):
    #     kernel_v[0][i] = 1
    kernel_v /= kernel_size
    # kernel_h /= kernel_size

    # Apply the vertical kernel.
    mb = cv2.filter2D(img, -1, kernel_v)

    return mb


if __name__ == '__main__':
    image_address = f"E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\images\Capture48.PNG"
    img = cv2.imread(image_address)
    new_img = calhe_filter_with_lab(img)
    # new_img = convert_scale_abs(img)
    # cv2.imshow('original', img)
    # cv2.imshow('adjusted', new_img)

    mb = apply_motion_blur(img, kernel_size=80)
    cv2.imshow('motion blurred', resize(mb,max_size=1500))
    while cv2.waitKey() != ord('q'):
        continue
    cv2.destroyAllWindows()
