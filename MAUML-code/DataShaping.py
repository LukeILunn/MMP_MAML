import pickle
import itertools
import math

with open("pickle_files/sittingFeatures", 'rb') as f:
    sitting_data = pickle.load(f)

with open("pickle_files/standingFeatures", 'rb') as f:
    standing_data = pickle.load(f)

with open("pickle_files/walkingFeatures", 'rb') as f:
    walking_data = pickle.load(f)

with open("pickle_files/redSittingFeatures", 'rb') as f:
    red_sitting_data = pickle.load(f)

with open("pickle_files/redStandingFeatures", 'rb') as f:
    red_standing_data = pickle.load(f)

with open("pickle_files/redWalkingFeatures", 'rb') as f:
    red_walking_data = pickle.load(f)

num_feature_lists = len(sitting_data[0])
num_examples = len(sitting_data[0][0])
num_markers = len(sitting_data)

red_num_examples = len(red_sitting_data[0])
red_num_features = len(red_sitting_data[0][0])

target = []
red_target = []

def build_list_shape(this_list, this_target):
    global target, num_markers, num_feature_lists, num_examples
    marker_concatenated = []
    all_markers_joined = []
    data_processed = []

    for i, j, k in itertools.product(range(0, num_examples), range(0, num_markers), range(0, num_feature_lists)):

        marker_concatenated += this_list[j][k][i]

        if k == 2:
            all_markers_joined.append(marker_concatenated)
            marker_concatenated = []

            if j == num_markers-1:
                data_processed.append(all_markers_joined)
                target.append(this_target)
                all_markers_joined = []

    return data_processed


def build_red_list_shape(this_list, this_target):
    global red_target, red_num_examples, red_num_features
    markers_concatenated = []
    data_processed = []

    for i in range(0, red_num_examples):
        markers_concatenated = this_list[0][i] + this_list[1][i]
        data_processed.append(markers_concatenated)
        markers_concatenated = []
        red_target.append(this_target)

    return data_processed


data = build_list_shape(sitting_data, "sitting")
data += build_list_shape(standing_data, "standing")
data += build_list_shape(walking_data, "walking")

with open("pickle_files/movement_data", 'wb') as f:
    pickle.dump(data, f)
    f.close()

with open("pickle_files/movement_targets", 'wb') as fi:
    pickle.dump(target, fi)
    fi.close()

red_data = build_red_list_shape(red_sitting_data, "sitting")
red_data += build_red_list_shape(red_standing_data, "standing")
red_data += build_red_list_shape(red_walking_data, "walking")

print(*red_data[0], sep='\n')

with open("pickle_files/red_movement_data", 'wb') as f:
    pickle.dump(red_data, f)
    f.close()

with open("pickle_files/red_movement_targets", 'wb') as fi:
    pickle.dump(red_target, fi)
    fi.close()


