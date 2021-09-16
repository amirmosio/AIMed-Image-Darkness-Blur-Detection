import cv2
import sys  # System bindings
import cv2  # OpenCV bindings
import numpy as np
from collections import Counter


class BackgroundColorDetector():
    # https://medium.com/generalist-dev/background-colour-detection-using-opencv-and-python-22ed8655b243
    def __init__(self, img, sample_percent=5.0):
        self.img = img
        self.sample_percent = sample_percent
        self.manual_count = {}
        self.w, self.h, self.channels = self.img.shape
        self.total_pixels = self.w * self.h

    def count(self):
        for y in range(0, self.h):
            for x in range(0, self.w):
                RGB = (self.img[x, y, 0], self.img[x, y, 1], self.img[x, y, 2])
                if RGB in self.manual_count:
                    self.manual_count[RGB] += 1
                else:
                    self.manual_count[RGB] = 1

    def average_colour(self):
        most_counts = int(len(self.manual_count) * (self.sample_percent / 100))
        number_counter = Counter(self.manual_count).most_common(most_counts)
        red = 0
        green = 0
        blue = 0
        for top in range(len(number_counter)):
            red += number_counter[top][0][0]
            green += number_counter[top][0][1]
            blue += number_counter[top][0][2]

        average_red = red / len(number_counter)
        average_green = green / len(number_counter)
        average_blue = blue / len(number_counter)
        return average_red, average_green, average_blue

    def detect(self):
        self.count()
        return self.average_colour()

    def show_back_ground_colour(self):
        c = self.detect()
        cv2.imshow('original', self.img)
        bgc = self.img.copy()
        for i in range(bgc.shape[0]):
            for j in range(bgc.shape[1]):
                bgc[i][j] = c
        cv2.imshow('background', bgc)


if __name__ == '__main__':
    image_address = f"E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\images\index2.jpg"
    img = cv2.imread(image_address)
    background_color = BackgroundColorDetector(img, sample_percent=0.5)

    background_color.show_back_ground_colour()
    while cv2.waitKey() != ord('q'):
        continue
    cv2.destroyAllWindows()
