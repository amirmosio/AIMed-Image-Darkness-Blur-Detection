import os
import time

import cv2
import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from BlurrDetection2.feature_class import FeatureValue
from BlurrDetection2.fft_mean_magnitude import FFTMeanMagnitude
from BlurrDetection2.laplacian import LaplacianOnWhole
from BlurrDetection2.laplacian_with_text_segmentation import LaplacianWithTextSegmentation
from BlurrDetection2.lpss import LPSS
from BlurrDetection2.max_saturation import MaxSaturation
from BlurrDetection2.walvet import WavletTransform

base_dir = os.path.split(os.getcwd())[0]
blur_detection_folder_path = os.getcwd()
debug = False


class BlurDetection:
    def __init__(self, features):
        self.features: [FeatureValue] = features
        self.classifier = None

    def set_classifier(self, classifier):
        self.classifier = classifier

    def feature_vector(self, img):
        feature_values = [feature(img).calculate_value() for feature in self.features]

        return np.array(feature_values)

    def train(self, x_train, y_train):
        # x_train = np.array([self.feature_vector(images[i]) for i in range(len(images))])
        # y_train = np.array([(1 if (labels[i] == 'Blur') else 0) for i in range(len(labels))])
        self.classifier.fit(x_train, y_train)

    def predict(self, x_test, test_labels):
        # x_test = np.array([self.feature_vector(test_images[i]) for i in range(len(test_images))])
        # y_test = np.array([(1 if (test_labels[i] == 'Blur') else 0) for i in range(len(test_labels))])
        predictions = self.classifier.predict(x_test)
        acc = accuracy_score(predictions, test_labels)
        return predictions, acc


def archive_dataset_data(feature_value_extractor):
    dataset_path = os.path.join(base_dir, "Data", "archive")
    total_image_features = []
    total_labels = []

    for group in os.listdir(dataset_path):
        group_folder_path = os.path.join(dataset_path, group)
        for i, file_name in enumerate(os.listdir(group_folder_path)):
            img_address = os.path.join(group_folder_path, file_name)
            img = cv2.imread(img_address)
            feature_values = feature_value_extractor(img)
            total_image_features.append(feature_values)
            if group == "sharp":
                total_labels.append('Sharp')
            else:
                total_labels.append('Blur')
    return np.array(total_image_features), np.array(total_labels)


def motion_blur_text_dataset_Data(feature_value_extractor):
    dataset_path = os.path.join(base_dir, "Data", "Motion Blur Text Images")
    total_image_features = []
    total_labels = []
    for group in os.listdir(dataset_path):
        group_folder_path = os.path.join(dataset_path, group)
        for i, file_name in enumerate(os.listdir(group_folder_path)):
            img_address = os.path.join(group_folder_path, file_name)
            img = cv2.imread(img_address)
            feature_values = feature_value_extractor(img)
            total_image_features.append(feature_values)
            if group == "resize":
                total_labels.append('Sharp')
            else:
                total_labels.append('Blur')
    return np.array(total_image_features), np.array(total_labels)


def custom_image_folder_data(feature_value_extractor):
    dataset_path = os.path.join(base_dir, "Images", )
    total_image_features = []
    total_labels = []

    image_labels = ["Sharp" for i in range(61)]
    blur_image_numbers = [8, 12, 14, 17, 20, 21, 23, 24, 25, 26, 27, 28, 32, 34, 38, 39, 40, 43, 45, 47, 49, 52, 53, 56,
                          58, 59, 60, 61]
    for i in blur_image_numbers:
        image_labels[i - 1] = "Blur"

    for i, file_name in enumerate(os.listdir(dataset_path)):
        img_address = os.path.join(dataset_path, file_name)
        img = cv2.imread(img_address)
        feature_values = feature_value_extractor(img)
        total_image_features.append(feature_values)
        file_index = int(file_name.split(".")[0][7:]) - 1
        total_labels.append(image_labels[file_index])
    return np.array(total_image_features), np.array(total_labels)


def load_dataset_x_y_values(feature_value_extractor):
    x_archive, y_archive = archive_dataset_data(feature_value_extractor)
    x_motion_blur, y_motion_blur = motion_blur_text_dataset_Data(feature_value_extractor)
    x_custom, y_custom = custom_image_folder_data(feature_value_extractor)
    x_data = x_custom + x_archive + x_motion_blur
    y_data = y_custom + y_archive + y_motion_blur
    return x_data, y_data


def main():
    pkl_folder_path = os.path.join(blur_detection_folder_path, "classifiers_results_pkl")
    load = False
    if load:
        # TODO
        pkl_file_name = "blur_detection.joblib.pkl"
        blurry_detection = joblib.load(os.path.join(pkl_folder_path, pkl_file_name))

        # DO stuff
    else:
        # training
        classifiers = [
            SVC(kernel='rbf'),
            SVC(gamma=2, C=1),
            SVC(kernel="linear", C=0.025),
            GaussianNB(),
            GaussianProcessClassifier(1.0 * RBF(1.0))
        ]

        features = [FFTMeanMagnitude, LaplacianOnWhole, LaplacianWithTextSegmentation, LPSS,
                    MaxSaturation, WavletTransform]

        blur_detection = BlurDetection(features)

        # gathering images
        print("load all images data")
        total_image_features, total_labels = load_dataset_x_y_values(blur_detection.feature_vector)
        train_x, test_x, train_y, test_y = train_test_split(total_image_features, total_labels,
                                                            test_size=0.3,
                                                            random_state=42)
        # saving calculated total_image_features and total labels
        image_features = os.path.join(pkl_folder_path,
                                      f"all_images_features-{total_image_features.shape[0]}.joblib.pkl")
        _ = joblib.dump(total_image_features, image_features, compress=9)
        image_labels = os.path.join(pkl_folder_path, f"all_images_labels-{total_labels.shape[0]}.joblib.pkl")
        _ = joblib.dump(total_labels, image_labels, compress=9)

        for i, cls in enumerate(classifiers):
            blur_detection.set_classifier(cls)
            file_address = os.path.join(pkl_folder_path, f"blur_detection{i}.joblib.pkl")
            print(f"training {i} started ")
            start_training_time = time.time()
            blur_detection.train(train_x, train_y)
            print(f"training {i} ended in {time.time() - start_training_time}")

            _ = joblib.dump(blur_detection, file_address, compress=9)

            print(f'testing {i} started')
            print(blur_detection.predict(test_x, test_y)[1])


if __name__ == '__main__':
    main()
