import cv2
import numpy as np

from Utils.resize_img_with_bound import resize


def channel_l(image, dim=10):
    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    # Normalize L channel by dividing all pixel values with maximum pixel value
    L = L / np.max(L)
    # Return True if mean is greater than thresh else False
    return np.mean(L)


# main function
def l_channel_light_check(img):
    threshold = 0.5
    img = resize(img)
    v = channel_l(img)
    return v >= threshold, v


if __name__ == '__main__':
    darkness_threshold = 90  # you need to determine what threshold to use
    for i in range(1, 32):
        image_address = 'E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\images\Capture' + str(i) + '.PNG'
        img = cv2.imread(image_address)
        print(i)
        print(l_channel_light_check(img))
    while cv2.waitKey() != ord('q'):
        continue
    cv2.destroyAllWindows()
