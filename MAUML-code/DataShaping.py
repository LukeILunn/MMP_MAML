import pickle
import itertools
import numpy as np

with open("pickle_files/FeatureExtraction_results/sittingFeatures", 'rb') as f:
    sitting_data = pickle.load(f)
    f.close()

with open("pickle_files/FeatureExtraction_results/standingFeatures", 'rb') as f:
    standing_data = pickle.load(f)
    f.close()

with open("pickle_files/FeatureExtraction_results/walkingFeatures", 'rb') as f:
    walking_data = pickle.load(f)
    f.close()

with open("pickle_files/FeatureExtraction_results/redSittingFeatures", 'rb') as f:
    red_sitting_data = pickle.load(f)
    f.close()

with open("pickle_files/FeatureExtraction_results/redStandingFeatures", 'rb') as f:
    red_standing_data = pickle.load(f)
    f.close()

with open("pickle_files/FeatureExtraction_results/redWalkingFeatures", 'rb') as f:
    red_walking_data = pickle.load(f)
    f.close()

num_feature_lists = len(sitting_data[0])
num_sat_examples = len(sitting_data[0][0])
num_stood_examples = len(standing_data[0][0])
num_walk_examples = len(walking_data[0][0])
num_markers = len(sitting_data)

red_num_sat_examples = len(red_sitting_data[0])
red_num_stood_examples = len(red_standing_data[0])
red_num_walk_examples = len(red_walking_data[0])
red_num_features = len(red_sitting_data[0][0])

target = []
red_target = []


def build_list_shape(this_list, this_target, these_examples):
    global target, num_markers, num_feature_lists
    marker_concatenated = []
    all_markers_joined = []
    data_processed = []

    for i, j, k in itertools.product(range(these_examples), range(0, num_markers), range(0, num_feature_lists)):

        variable = this_list[j][k][i]
        marker_concatenated += variable

        if k == 2:
            all_markers_joined.append(marker_concatenated)
            marker_concatenated = []

            if j == num_markers-1:
                nan_check = np.asarray(all_markers_joined)
                if not np.isnan(nan_check).any():
                    data_processed.append(all_markers_joined)
                    target.append(this_target)
                all_markers_joined = []

    return data_processed


def build_red_list_shape(this_list, this_target, these_examples):
    global red_target, red_num_features
    data_processed = []

    for i in range(0, these_examples):
        markers_concatenated = this_list[0][i] + this_list[1][i]
        nan_check = np.asarray(markers_concatenated)
        if not np.isnan(nan_check).any():
            data_processed.append(markers_concatenated)
            red_target.append(this_target)

    return data_processed


data = build_list_shape(sitting_data, "sitting", num_sat_examples)
data += build_list_shape(standing_data, "standing", num_stood_examples)
data += build_list_shape(walking_data, "walking", num_walk_examples)

with open("pickle_files/DataShaping_results/movement_data", 'wb') as f:
    pickle.dump(data, f)
    f.close()

with open("pickle_files/DataShaping_results/movement_targets", 'wb') as fi:
    pickle.dump(target, fi)
    fi.close()

red_data = build_red_list_shape(red_sitting_data, "sitting", red_num_sat_examples)
red_data += build_red_list_shape(red_standing_data, "standing", red_num_stood_examples)
red_data += build_red_list_shape(red_walking_data, "walking", red_num_walk_examples)

with open("pickle_files/DataShaping_results/red_movement_data", 'wb') as f:
    pickle.dump(red_data, f)
    f.close()

with open("pickle_files/DataShaping_results/red_movement_targets", 'wb') as fi:
    pickle.dump(red_target, fi)
    fi.close()
