import cv2

from BlurrDetection.feature_class import FeatureValue


class MaxSaturation(FeatureValue):
    def __init__(self, img):
        super().__init__(img, "max saturation")
        self.img = img

    def max_saturation(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img_hsv[:, :, 1].max()

    def calculate_value(self):
        return self.max_saturation(self.img)
