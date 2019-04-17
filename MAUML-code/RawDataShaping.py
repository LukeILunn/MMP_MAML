import pickle
import itertools

with open("pickle_files/redSittingData", 'rb') as f:
    red_sitting_data = pickle.load(f)

with open("pickle_files/redStandingData", 'rb') as f:
    red_standing_data = pickle.load(f)

with open("pickle_files/redWalkingData", 'rb') as f:
    red_walking_data = pickle.load(f)
    f.close()

num_markers = 2
num_frames = 50
num_examples = len(red_walking_data[0])
data_joined = []
data = []
targets = []


for i in range(0, num_examples):
    red_sitting_data[0][i] = red_sitting_data[0][i].tolist()
    red_sitting_data[1][i] = red_sitting_data[1][i].tolist()
    data_joined.append(red_sitting_data[0][i] + red_sitting_data[1][i])
    targets.append("sitting")


for i in range(0, num_examples):
    red_standing_data[0][i] = red_standing_data[0][i].tolist()
    red_standing_data[1][i] = red_standing_data[1][i].tolist()
    data_joined.append(red_standing_data[0][i] + red_standing_data[1][i])
    targets.append("standing")

for i in range(0, num_examples):
    red_walking_data[0][i] = red_walking_data[0][i].tolist()
    red_walking_data[1][i] = red_walking_data[1][i].tolist()
    data_joined.append(red_walking_data[0][i] + red_walking_data[1][i])
    targets.append("walking")


print(len(data_joined[0]))

with open("pickle_files/raw_data_reduced_set", 'wb') as f:
    pickle.dump(data_joined, f)
    f.close()

with open("pickle_files/raw_data_reduced_set_targets", 'wb') as f:
    pickle.dump(targets, f)
    f.close()