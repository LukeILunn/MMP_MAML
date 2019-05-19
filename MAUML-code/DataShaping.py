import pickle
import itertools
import numpy as np

# Files created during FeatureExtraction.py are loaded into the script.
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

# Variables here record data about the dimensions of the data stored
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

# This method is designed to take the feature lists and shape them such that input[n] will contain
# a feature list from each marker corresponding to time step n.
def build_list_shape(this_list, this_target, these_examples):
    print("Joining all " + this_target + "feature lists together.\n")
    global target, num_markers, num_feature_lists
    marker_concatenated = []
    all_markers_joined = []
    data_processed = []

    for i, j, k in itertools.product(range(these_examples), range(0, num_markers), range(0, num_feature_lists)):
        # concatenate the 3 different types of value from a single marker
        variable = this_list[j][k][i]
        marker_concatenated += variable

        if k == 2:
            # append the marker to the full marker set
            all_markers_joined.append(marker_concatenated)
            marker_concatenated = []

            if j == num_markers-1:
                nan_check = np.asarray(all_markers_joined)
                if not np.isnan(nan_check).any():
                    # append the full marker set with all data types to data_processed
                    data_processed.append(all_markers_joined)
                    target.append(this_target)
                all_markers_joined = []

    return data_processed

# Performs the same process as the method above but for the reduced data set which only uses a single data
# type (acceleration) and only 2 markers, making it much shorter.
def build_red_list_shape(this_list, this_target, these_examples):
    print("Joining all " + this_target + "feature lists for the reduced set.\n")
    global red_target, red_num_features
    data_processed = []

    for i in range(0, these_examples):
        markers_concatenated = this_list[0][i] + this_list[1][i]
        nan_check = np.asarray(markers_concatenated)
        if not np.isnan(nan_check).any():
            data_processed.append(markers_concatenated)
            red_target.append(this_target)

    return data_processed

# Here the data shaping method is called for all of the 3 data recordings, such that all
# exist within a single list, and have a corresponding list of targets, which are the ground truths
data = build_list_shape(sitting_data, "sitting", num_sat_examples)
data += build_list_shape(standing_data, "standing", num_stood_examples)
data += build_list_shape(walking_data, "walking", num_walk_examples)

# Storing the result of DataShaping.py to be passed to ClassifierTesting.py
with open("pickle_files/DataShaping_results/movement_data", 'wb') as f:
    pickle.dump(data, f)
    f.close()

with open("pickle_files/DataShaping_results/movement_targets", 'wb') as fi:
    pickle.dump(target, fi)
    fi.close()

# Performs the same process as above for the reduced set.
red_data = build_red_list_shape(red_sitting_data, "sitting", red_num_sat_examples)
red_data += build_red_list_shape(red_standing_data, "standing", red_num_stood_examples)
red_data += build_red_list_shape(red_walking_data, "walking", red_num_walk_examples)

with open("pickle_files/DataShaping_results/red_movement_data", 'wb') as f:
    pickle.dump(red_data, f)
    f.close()

with open("pickle_files/DataShaping_results/red_movement_targets", 'wb') as fi:
    pickle.dump(red_target, fi)
    fi.close()

print("Data successfully shaped, please run RawDataShaping.py if ou haven't already, "
      "then move on to ClassifierTesting.py\n")
