import pickle as pick
from scipy.stats import kurtosis, skew
import numpy as np
import itertools


def extract_features(this_list):
    this_list = np.array(this_list)
    x_max, y_max, z_max = np.amax(this_list, axis=0)
    x_min, y_min, z_min = np.amin(this_list, axis=0)
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    x_mean, y_mean, z_mean = np.mean(this_list, axis=0)
    x_variance, y_variance, z_variance = np.var(this_list, axis=0)
    x_stdev, y_stdev, z_stdev = np.std(this_list, axis=0)
    x_kurt, y_kurt, z_kurt = kurtosis(this_list, axis=0)
    x_skew, y_skew, z_skew = skew(this_list, axis=0)
    feature_list = [x_max, y_max, z_max, x_min, y_min, z_min,
                    x_range, y_range, z_range, x_mean, y_mean, z_mean,
                    x_variance, y_variance, z_variance, x_stdev, y_stdev, z_stdev,
                    x_kurt, y_kurt, z_kurt, x_skew, y_skew, z_skew]
    return feature_list


with open("pickle_files/DataExtraction_results/stoodData", 'rb') as f:
    stood_data = pick.load(f)
    f.close()

with open("pickle_files/DataExtraction_results/satData", 'rb') as f:
    sat_data = pick.load(f)
    f.close()

with open("pickle_files/DataExtraction_results/walkingData", 'rb') as f:
    walking_data = pick.load(f)
    f.close()

with open("pickle_files/DataExtraction_results/redStandingData", 'rb') as f:
    red_stood_data = pick.load(f)
    f.close()

with open("pickle_files/DataExtraction_results/redSittingData", 'rb') as f:
    red_sat_data = pick.load(f)
    f.close()

with open("pickle_files/DataExtraction_results/redWalkingData", 'rb') as f:
    red_walking_data = pick.load(f)
    f.close()

marker_type = len(walking_data)
red_marker_type = len(red_walking_data)
set_type = len(walking_data[0])
set_number = max(len(walking_data[0][0]), len(stood_data[0][0]), len(sat_data[0][0]))

standing_features = [[[0 for k in range(len(stood_data[0][0]))]for j in range(set_type)] for i in range(marker_type)]
sitting_features = [[[0 for k in range(len(sat_data[0][0]))]for j in range(set_type)] for i in range(marker_type)]
walking_features = [[[0 for k in range(len(walking_data[0][0]))]for j in range(set_type)] for i in range(marker_type)]

red_standing_features = [[0 for k in range(len(red_stood_data[0]))]for i in range(red_marker_type)]
red_sitting_features = [[0 for k in range(len(red_sat_data[0]))]for i in range(red_marker_type)]
red_walking_features = [[0 for k in range(len(red_walking_data[0]))]for i in range(red_marker_type)]


for i, j, k in itertools.product(range(0, marker_type), range(0, set_type), range(0, set_number)):
    if k < len(stood_data[0][0]):
        standing_features[i][j][k] = extract_features(stood_data[i][j][k])

    if k < len(sat_data[0][0]):
        sitting_features[i][j][k] = extract_features(sat_data[i][j][k])

    if k < len(walking_data[0][0]):
        walking_features[i][j][k] = extract_features(walking_data[i][j][k])


for i, k in itertools.product(range(0, red_marker_type), range(0, set_number)):
    if k < len(red_stood_data[0]):
        red_standing_features[i][k] = extract_features(red_stood_data[i][k])

    if k < len(red_sat_data[0]):
        red_sitting_features[i][k] = extract_features(red_sat_data[i][k])

    if k < len(red_walking_data[0]):
        red_walking_features[i][k] = extract_features(red_walking_data[i][k])


pickle_out = open("pickle_files/FeatureExtraction_results/walkingFeatures", 'wb')
pick.dump(walking_features, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/FeatureExtraction_results/standingFeatures", 'wb')
pick.dump(standing_features, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/FeatureExtraction_results/sittingFeatures", 'wb')
pick.dump(sitting_features, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/FeatureExtraction_results/redWalkingFeatures", 'wb')
pick.dump(red_walking_features, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/FeatureExtraction_results/redStandingFeatures", 'wb')
pick.dump(red_standing_features, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/FeatureExtraction_results/redSittingFeatures", 'wb')
pick.dump(red_sitting_features, pickle_out)
pickle_out.close()
