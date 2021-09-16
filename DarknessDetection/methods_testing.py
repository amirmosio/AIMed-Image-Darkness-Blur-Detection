import os
import time

import cv2

from DarknessDetection.darkness_detection import DarknessDetection
from DarknessDetection.resize_img_with_bound import resize_with_scale


def run_on_brightening_dataset(img_scale=1):
    archive_folder = f'E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\Data\BrighteningTrain'
    high_folder_address = archive_folder + "\\" + os.listdir(archive_folder)[0]
    error_count = 0
    total = sum([len(os.listdir(archive_folder + "\\" + group)) for group in os.listdir(archive_folder)])
    start_time = time.time()
    for i in range(len(os.listdir(high_folder_address))):
        for group in os.listdir(archive_folder):
            group_folder = archive_folder + "\\" + group
            img_address = group_folder + '\\' + os.listdir(group_folder)[i]
            dict = DarknessDetection()(resize_with_scale(cv2.imread(img_address), scale=img_scale))
            b, v = dict['check'], dict['value']
            if (b and group != 'high') or (not b and group == 'high'):
                # print(group, i)
                # print("**************error with " + str(v) + "*************")
                # print(img_address)
                error_count += 1
    print(time.time() - start_time)
    print("total image:", total)
    print("error count:", error_count)


def run_on_lol_dataset(img_scale=1):
    archive_folder = f'E:\Documentwork\sharif\CE Project\Prev\Anomaly Detection - Phase1\Data\our485'
    high_folder_address = archive_folder + "\\" + os.listdir(archive_folder)[0]
    error_count = 0
    total = sum([len(os.listdir(archive_folder + "\\" + group)) for group in os.listdir(archive_folder)])
    start_time = time.time()
    for i in range(len(os.listdir(high_folder_address))):
        for group in os.listdir(archive_folder):
            group_folder = archive_folder + "\\" + group
            img_address = group_folder + '\\' + os.listdir(group_folder)[i]
            dict = DarknessDetection()(resize_with_scale(cv2.imread(img_address), scale=img_scale))
            b, v = dict['check'], dict['value']
            if (b and group != 'high') or (not b and group == 'high'):
                # print(group, i)
                # print("**************error with " + str(v) + "*************")
                # print(img_address)
                error_count += 1
    print(time.time() - start_time)
    print("total image:", total)
    print("error count:", error_count)


if __name__ == '__main__':
    for i in [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        print(i)
        print("brightening ...")
        run_on_brightening_dataset(img_scale=i)
        print("lol ...")
        run_on_lol_dataset(img_scale=i)


# 1
# brightening ...
# 13.502018451690674
# total image: 2000
# error count: 296
# lol ...
# 33.532166957855225
# total image: 970
# error count: 50
# 0.9
# brightening ...
# 13.180015802383423
# total image: 2000
# error count: 301
# lol ...
# 9.276375770568848
# total image: 970
# error count: 59
# 0.8
# brightening ...
# 12.53117299079895
# total image: 2000
# error count: 298
# lol ...
# 8.238073587417603
# total image: 970
# error count: 61
# 0.7
# brightening ...
# 11.64331603050232
# total image: 2000
# error count: 302
# lol ...
# 8.242728233337402
# total image: 970
# error count: 57
# 0.6
# brightening ...
# 10.963865041732788
# total image: 2000
# error count: 299
# lol ...
# 7.386747598648071
# total image: 970
# error count: 53
# 0.5
# brightening ...
# 10.756165027618408
# total image: 2000
# error count: 302
# lol ...
# 6.987260103225708
# total image: 970
# error count: 58
# 0.4
# brightening ...
# 10.208602905273438
# total image: 2000
# error count: 294
# lol ...
# 6.691268444061279
# total image: 970
# error count: 51
# 0.3
# brightening ...
# 17.389931678771973
# total image: 2000
# error count: 293
# lol ...
# 10.405868530273438
# total image: 970
# error count: 48
# 0.2
# brightening ...
# 12.234347820281982
# total image: 2000
# error count: 301
# lol ...
# 6.885379314422607
# total image: 970
# error count: 47
# 0.1
# brightening ...
# 11.119303464889526
# total image: 2000
# error count: 304
# lol ...
# 6.451562881469727
# total image: 970
# error count: 63