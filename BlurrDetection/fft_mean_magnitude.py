import math

import cv2
import numpy as np

from BlurrDetection.feature_class import FeatureValue


class FFTMeanMagnitude(FeatureValue):
    def __init__(self, img):
        super().__init__(img, "fft mean magnitude")

    def mean_magnitude(self, img, size=50):
        # grab the dimensions of the image and use the dimensions to
        # derive the center (x, y)-coordinates
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (h, w) = image.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
        # compute the FFT to find the frequency transform, then shift
        # the zero frequency component (i.e., DC component located at
        # the top-left corner) to the center where it will be more

        fft = np.fft.fft2(image)
        fftShift = np.fft.fftshift(fft)
        fftShift[cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        # the image will be considered "blurry" if the mean value of the
        # magnitudes is less than the threshold value
        if mean == -1 * math.inf:
            return 99999999
        return mean

    def calculate_value(self):
        return self.mean_magnitude(self.img)
