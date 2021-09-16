import cv2

from DarknessDetection.grayscale_laplacian_with_v_channel import readability_and_light_check
from DarknessDetection.v_channel_mean import v_channel_light_check


# import wize

class DarknessDetection:
    def __init__(self):
        pass

    def __call__(self, img):
        # image_url = starlette_request.json['url']
        # req = urlopen(image_url)
        # arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # img = cv2.imdecode(arr, -1)
        # if len(img.shape) != 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # cv2.imshow("t",img)
        c1, v1 = readability_and_light_check(img)
        c2, v2 = v_channel_light_check(img)
        # return {'check': bool(v1 / 50 + v2 / 90 > 2), 'value': v1 / 50 + v2 / 90}
        return {'check': c1 and c2, 'value': v1 / 50 + v2 / 90}


if __name__ == '__main__':
    threshold = 50
    for i in range(52, 53):
        image_address = 'E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\images\Capture' + str(i) + '.PNG'
        img = cv2.imread(image_address)
        print(i)
        print(DarknessDetection()(img))

    while cv2.waitKey() != ord('q'):
        continue
    cv2.destroyAllWindows()
