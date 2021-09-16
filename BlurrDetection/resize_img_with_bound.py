import cv2
# import wize

def resize(img, max_size=800):
    img.resize()
    scale = max_size / (max(img.shape)+0.001)
    h_size = int(img.shape[1] * scale)
    w_size = int(img.shape[0] * scale)
    resized_img = cv2.resize(img, (h_size, w_size))
    return resized_img
