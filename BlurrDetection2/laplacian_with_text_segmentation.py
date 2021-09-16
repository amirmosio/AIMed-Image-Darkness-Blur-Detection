import random as rand

import cv2
import numpy as np

from BlurrDetection2.feature_class import FeatureValue
from BlurrDetection2.text_segmentation import maximality_region_detection


class LaplacianWithTextSegmentation(FeatureValue):
    def __init__(self, img):
        super().__init__(img, "laplacian-with-segmentation")
        self.img = img

    def _grayscale_laplacian_with_mean_v_channel(self, img_region):
        gray = cv2.cvtColor(img_region, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

        std = cv2.cv2.meanStdDev(laplacian)[1][0][0]
        max_value = cv2.minMaxLoc(laplacian)[1]

        variance = std ** 2

        return max_value, variance

    def _cut_text_segmentation(self, img, points):
        img_y, img_x = img.shape[0], img.shape[1]
        mask = np.zeros(img.shape[0:2], dtype=np.uint8)
        # method 1 smooth region

        # adding extra 20 pixel for margin
        extra_mating = 0
        avgx1 = np.average(points[:, 0])
        avgy1 = np.average(points[:, 1])
        width = np.max(points[:, 0]) - np.min(points[:, 0]) + extra_mating
        height = np.max(points[:, 1]) - np.min(points[:, 1]) + extra_mating

        points = np.asarray(
            [[(avgx1 - width / 2, avgy1 - height / 2)], [(avgx1 - width / 2, avgy1 + height / 2)],
             [(avgx1 + width / 2, avgy1 + height / 2)], [(avgx1 + width / 2, avgy1 - height / 2)]])

        points = np.round(points).astype(int)
        points[:, :, 0] = np.clip(points[:, :, 0], a_min=0, a_max=img_x - 1)
        points[:, :, 1] = np.clip(points[:, :, 1], a_min=0, a_max=img_y - 1)
        # cropping img
        rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
        cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        res = cv2.bitwise_and(img, img, mask=mask)

        cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
        return cropped

    def segment_blurry_check(self, img, sample_count=200):
        threshold = 1950
        hulls = maximality_region_detection(img)
        stds = np.array([])
        for i, hull in enumerate(rand.sample(hulls, min(int(len(hulls)), sample_count))):
            cropped_img = self._cut_text_segmentation(img, hull)
            max_val, variance = self._grayscale_laplacian_with_mean_v_channel(cropped_img)
            stds = np.append(stds, variance)
        stds.sort()
        if len(stds) == 0:
            return False, 0, (0, 0)
        return sum(stds) / len(stds) > threshold, sum(stds) / len(stds), (min(stds), max(stds))

    def calculate_value(self):
        return self.segment_blurry_check(self.img)[1]
