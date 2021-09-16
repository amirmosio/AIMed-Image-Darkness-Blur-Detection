class FeatureValue:
    def __init__(self, img, feature_name):
        self.img = img
        self.feature_name = feature_name

    def calculate_value(self):
        raise Exception("Not implemented")
