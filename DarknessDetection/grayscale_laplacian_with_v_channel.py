import cv2
# import wize

def grayscale_laplacian_with_mean_v_channel_value(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    gray_mean, _, _, _ = cv2.mean(gray)
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(laplacian)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_mean, _, _, _ = cv2.mean(hsv_img[:, :, 2])

    return (max_value * v_mean) / (gray_mean+0.001)


# main function
def readability_and_light_check(img):
    threshold = 50
    # TODO with or without resize
    # img = resize(img)
    v = grayscale_laplacian_with_mean_v_channel_value(img)
    return v >= threshold, v


# if __name__ == '__main__':
#     threshold = 50
#     for i in range(1, 32):
#         image_address = 'E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\images\Capture' + str(i) + '.PNG'
#         img = cv2.imread(image_address)
#         print(i)
#         print(readability_and_light_check(img))
#
#     while cv2.waitKey() != ord('q'):
#         continue
#     cv2.destroyAllWindows()
