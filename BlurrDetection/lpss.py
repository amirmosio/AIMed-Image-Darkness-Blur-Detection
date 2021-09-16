import cv2
import numpy as np
import scipy.stats as stats
from scipy.stats import linregress

from BlurrDetection.feature_class import FeatureValue


class LPSS(FeatureValue):
    def __init__(self, img):
        super().__init__(img, "LPSS")

    def lpss(self, img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        npix = image.shape[0]
        npiy = image.shape[1]

        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2

        kfreqx = np.fft.fftfreq(npix) * npix
        kfreqy = np.fft.fftfreq(npiy) * npiy
        kfreq2D = np.meshgrid(kfreqx, kfreqy)
        knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()

        kbins = np.arange(0.5, npix // 2 + 1, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                             statistic="mean",
                                             bins=kbins)
        Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        slope = -1 * linregress(np.log(kvals[0:-10]), np.log(Abins[0:-10])).slope
        if np.isnan(slope):
            return 99999999
        return slope

    def calculate_value(self):
        return self.lpss(self.img)
