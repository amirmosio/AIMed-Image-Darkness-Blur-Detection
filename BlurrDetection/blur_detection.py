from urllib.request import urlopen

import cv2
import numpy as np

from BlurDetection.laplacian_with_text_segmentation import segment_blurry_check


# import wize


class BlurDetection:
    def __init__(self):
        pass

    def __call__(self, img):
        # image_url = starlette_request.json['url']
        # req = urlopen(image_url)
        # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # img = cv2.imdecode(arr, -1)
        # if len(img.shape) != 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        b1, v1, extra = segment_blurry_check(img)
        return {'check': b1, 'value': v1}


if __name__ == '__main__':
    threshold = 50
    for i in range(51, 52):
        image_address = 'E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\images\Capture' + str(i) + '.PNG'
        img = cv2.imread(image_address)
        print(i)
        print(BlurDetection()(img))

    while cv2.waitKey() != ord('q'):
        continue
    cv2.destroyAllWindows()
