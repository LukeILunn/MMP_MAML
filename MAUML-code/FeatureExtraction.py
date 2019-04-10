import pickle as pick
import statistics as stat
import numpy as np


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
                    x_variance, y_variance, z_variance, x_stdev, y_stdev, z_stdev]
                    #covariance_matrix]
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
standing_targets = [[[0 for k in range(set_number)]for j in range(set_type)] for i in range(marker_type)]

sitting_features = [[[0 for k in range(set_number)]for j in range(set_type)] for i in range(marker_type)]
sitting_targets = [[[0 for k in range(set_number)]for j in range(set_type)] for i in range(marker_type)]

walking_features = [[[0 for k in range(set_number)]for j in range(set_type)] for i in range(marker_type)]
walking_targets = [[[0 for k in range(set_number)]for j in range(set_type)] for i in range(marker_type)]

standing_target = "standing"
sitting_target = "sitting"
walking_target = "walking"

for i in range(0, marker_type):
    for j in range(0, set_type):
        for k in range(0, set_number):
            standing_list_to_add = extract_features(stood_data[i][j][k])
            sitting_list_to_add = extract_features(sat_data[i][j][k])
            walking_list_to_add = extract_features(walking_data[i][j][k])
            standing_features[i][j][k] = standing_list_to_add
            sitting_features[i][j][k] = sitting_list_to_add
            walking_features[i][j][k] = walking_list_to_add
            standing_targets[i][j][k] = standing_target
            sitting_targets[i][j][k] = sitting_target
            walking_targets[i][j][k] = walking_target


pickle_out = open("pickle_files/walkingFeatures", 'wb')
pick.dump(walking_features, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/walkingTargets", 'wb')
pick.dump(walking_targets, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/standingFeatures", 'wb')
pick.dump(standing_features, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/standingTargets", 'wb')
pick.dump(standing_targets, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/sittingFeatures", 'wb')
pick.dump(sitting_features, pickle_out)
pickle_out.close()

pickle_out = open("pickle_files/sittingTargets", 'wb')
pick.dump(sitting_targets, pickle_out)
pickle_out.close()

print("this is a message, should have made files by now")
