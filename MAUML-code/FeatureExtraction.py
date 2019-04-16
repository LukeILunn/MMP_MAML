import pickle as pick
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
    #covariance_matrix = np.cov(this_list)
    feature_list = [x_max, y_max, z_max, x_min, y_min, z_min,
                    x_range, y_range, z_range, x_mean, y_mean, z_mean,
                    x_variance, y_variance, z_variance, x_stdev, y_stdev, z_stdev
                    #,covariance_matrix
                    ]
    return feature_list


with open("pickle_files/stoodData", 'rb') as f:
    stood_data = pick.load(f)

with open("pickle_files/satData", 'rb') as f:
    sat_data = pick.load(f)

with open("pickle_files/walkingData", 'rb') as f:
    walking_data = pick.load(f)

marker_type = len(walking_data)
set_type = len(walking_data[0])
set_number = len(walking_data[0][0])

standing_features = [[[0 for k in range(set_number)]for j in range(set_type)] for i in range(marker_type)]

sitting_features = [[[0 for k in range(set_number)]for j in range(set_type)] for i in range(marker_type)]

walking_features = [[[0 for k in range(set_number)]for j in range(set_type)] for i in range(marker_type)]

for i, j, k in itertools.product(range(0, marker_type), range(0, set_type), range(0, set_number)):
            standing_features[i][j][k] = extract_features(stood_data[i][j][k])
            sitting_features[i][j][k] = extract_features(sat_data[i][j][k])
            walking_features[i][j][k] = extract_features(walking_data[i][j][k])


pickle_out = open("pickle_files/walkingFeatures", 'wb')
pick.dump(walking_features, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/standingFeatures", 'wb')
pick.dump(standing_features, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/sittingFeatures", 'wb')
pick.dump(sitting_features, pickle_out)
pickle_out.close()

print("this is a message, should have made files by now")
