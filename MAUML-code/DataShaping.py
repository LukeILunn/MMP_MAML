import pickle
import itertools
import math

with open("pickle_files/sittingFeatures", 'rb') as f:
    sitting_data = pickle.load(f)

with open("pickle_files/standingFeatures", 'rb') as f:
    standing_data = pickle.load(f)

with open("pickle_files/walkingFeatures", 'rb') as f:
    walking_data = pickle.load(f)

num_markers = len(sitting_data)
num_feature_lists = len(sitting_data[0])
num_examples = len(sitting_data[0][0])

target = []


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

            if j == 15:
                data_processed.append(all_markers_joined)
                target.append(this_target)
                all_markers_joined = []

    return data_processed


data = build_list_shape(sitting_data, "sitting")
data += (build_list_shape(standing_data, "standing"))
data += (build_list_shape(walking_data, "walking"))

for i in range(len(data)):
    print(*data[i], sep='\n')
    print(" " '\n\n\n\n')

print(target[0])

with open("pickle_files/movement_data", 'wb') as f:
    pickle.dump(data, f)

with open("pickle_files/movement_targets", 'wb') as fi:
    pickle.dump(target, fi)
