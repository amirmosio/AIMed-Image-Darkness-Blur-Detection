import cv2
import numpy as np

from Utils.filter_and_editing import calhe_filter_with_lab
from Utils.resize_img_with_bound import resize


def non_max_suppression_fast(boxes, overlapThresh=0.1):
    # Empty array detection
    if len(boxes) == 0:
        return []

        # Convert the type to float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    # Four coordinate arrays
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)  # Calculate area array
    idxs = np.argsort(area)  # Returns the index value of the lower right corner coordinate from small to large
    idxs = np.flip(idxs)
    # Start traversing to delete duplicate boxes
    while len(idxs) > 0:
        # Put the bottom right box into the pick array
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest coordinate x1y1 and the smallest coordinate x2y2 in the remaining boxes,
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Calculate the proportion of overlapping area in the corresponding frame
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        # If the proportion is greater than the threshold, delete
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


def maximality_region_detection(img, delta=6, min_area=50, max_area=144000, max_variation=0.25, min_diversity=0.3,
                                min_margin=0.1):
    img = calhe_filter_with_lab(img)
    mser = cv2.MSER_create(_delta=delta, _min_area=min_area, _max_area=max_area, _max_variation=max_variation,
                           _min_diversity=min_diversity, _min_margin=min_margin)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    regions = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]

    keep = []
    # Draw the current rectangular text box
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        keep.append([x, y, x + w, y + h])
    keep2 = np.array(keep)
    pick = non_max_suppression_fast(keep2)
    hulls = []
    for (startX, startY, endX, endY) in pick:
        hulls.append(np.array([[startX, startY], [endX, startY], [endX, endY], [startX, endY]]))
    return hulls


def draw_detected_polylines(img, hulls, color=(0, 255, 0)):
    cv2.polylines(img, hulls, 1, color)
    # cv2.namedWindow('img', 0)
    cv2.imshow('img', resize(img))


if __name__ == '__main__':
    image_address = f'E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\images\Capture42.PNG'
    img = cv2.imread(image_address)
    img = resize(img)

    hulls = maximality_region_detection(img)
    print(len(hulls))
    draw_detected_polylines(img, hulls)
    while cv2.waitKey() != ord('q'):
        continue
    cv2.destroyAllWindows()
