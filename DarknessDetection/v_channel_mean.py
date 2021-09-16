import cv2
# import wize

def channel_v(img):
    img = img.copy()
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    v_mean, v_std = cv2.meanStdDev(v)

    return v_mean[0][0], v_std[0][0]


# main function
def v_channel_light_check(img):
    threshold = 90
    # img = resize(img)
    v = channel_v(img)[0]
    return v >= threshold, v


# if __name__ == '__main__':
#     darkness_threshold = 90  # you need to determine what threshold to use
#     for i in range(1, 32):
#         image_address = 'E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\images\Capture' + str(i) + '.PNG'
#         img = cv2.imread(image_address)
#         print(i)
#         print(v_channel_light_check(img))
#     while cv2.waitKey() != ord('q'):
#         continue
#     cv2.destroyAllWindows()
