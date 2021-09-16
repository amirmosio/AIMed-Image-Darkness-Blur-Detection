import os
import time

import cv2

from BlurDetection.laplacian import blurry_check
from BlurDetection.laplacian_with_text_segmentation import segment_blurry_check
from BlurDetection.resize_img_with_bound import resize


def run_on_archive_dataset():
    archive_folder = f'E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\Data\\archive'
    sharp_folder_address = (archive_folder + "\\" + os.listdir(archive_folder)[2]).replace("\\", "\\")
    error_count1 = {}
    error_count2 = {}
    error_count3 = {}
    error_count4 = {}
    total = sum([len(os.listdir(archive_folder + "\\" + group)) for group in os.listdir(archive_folder)])
    start_time = time.time()
    for i in range(len(os.listdir(sharp_folder_address))):
        for group in os.listdir(archive_folder):
            group_folder = archive_folder + "\\" + group
            img_address = group_folder + '\\' + os.listdir(group_folder)[i]
            img = cv2.imread(img_address)
            img = resize(img)
            # METHOD 1
            b1, v1, extra = segment_blurry_check(img)
            # METHOD 2
            b2, v2 = blurry_check(img)
            if (b1 and group != 'sharp') or (not b1 and group == 'sharp'):
                error_count1[group] = error_count1.get(group, 0) + 1
            if (b2 and group != 'sharp') or (not b2 and group == 'sharp'):
                error_count2[group] = error_count2.get(group, 0) + 1
            if (b2 and b1 and group != 'sharp') or ((not (b2 and b1)) and group == 'sharp'):
                error_count3[group] = error_count3.get(group, 0) + 1
            if b1 == b2 and ((b2 and b1 and group != 'sharp') or ((not (b2 and b1)) and group == 'sharp')):
                error_count4[group] = error_count4.get(group, 0) + 1
            print(i)

    print(time.time() - start_time)
    print("total image:", total)
    # METHOD 1
    print("error count1:", error_count1)
    # METHOD 2
    print("error count2:", error_count2)
    # Method 3
    print("error count3:", error_count3)
    print("error count4:", error_count4)


# def run_on_motion_blur_text_images():
#     archive_folder = f'E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\Data\\Motion Blur Text Images'
#     sharp_folder_address = (archive_folder + "\\" + os.listdir(archive_folder)[1]).replace("\\\\", "\\")
#     error_count = 0
#     total = sum([len(os.listdir(archive_folder + "\\" + group)) for group in os.listdir(archive_folder)])
#     start_time = time.time()
#     for i in range(1):
#         for group in os.listdir(archive_folder):
#             group_folder = archive_folder + "\\" + group
#             img_address = group_folder + '\\' + sorted(os.listdir(group_folder))[i]
#             img = cv2.imread(img_address)
#             cv2.imshow("sdgs", img)
#             b, v = segment_blurry_check(img, sample_count=50)
#             if (b and group != 'resize') or (not b and group == 'resize'):
#                 print(group, i)
#                 print("**************error with " + str(v) + "*************")
#                 print(img_address)
#                 error_count += 1
#     print(time.time() - start_time)
#     print("total image:", total)
#     print("error count:", error_count)


if __name__ == '__main__':
    # run_on_motion_blur_text_images()
    # print("motion blur text image done")
    run_on_archive_dataset()
