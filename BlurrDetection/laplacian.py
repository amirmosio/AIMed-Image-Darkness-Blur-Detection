import cv2

from BlurrDetection.feature_class import FeatureValue


class LaplacianOnWhole(FeatureValue):
    def __init__(self, img):
        super().__init__(img, "laplacian-on-whole-image")
        self.img = img

    def _grayscale_laplacian_with_mean_v_channel(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3, )
        std = cv2.cv2.meanStdDev(laplacian)[1][0][0]
        max_value = cv2.minMaxLoc(laplacian)[1]

        variance = std ** 2

        return max_value, variance

    def blurry_check(self, img):
        threshold = 600
        max_value, variance = self._grayscale_laplacian_with_mean_v_channel(img)
        return variance > threshold, variance

    def calculate_value(self):
        return self.blurry_check(self.img)[1]
