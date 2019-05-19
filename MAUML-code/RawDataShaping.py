import pickle
import numpy as np

# Loading in the raw data set from DataExtraction.py
with open("pickle_files/DataExtraction_results/redSittingData", 'rb') as f:
    red_sitting_data = pickle.load(f)

with open("pickle_files/DataExtraction_results/redStandingData", 'rb') as f:
    red_standing_data = pickle.load(f)

with open("pickle_files/DataExtraction_results/redWalkingData", 'rb') as f:
    red_walking_data = pickle.load(f)
    f.close()

# Storing information regarding the dimensions of the raw data
num_frames = len(red_standing_data[0][0])
num_sat_examples = len(red_sitting_data[0])
num_stood_examples = len(red_standing_data[0])
num_walk_examples = len(red_walking_data[0])
data_joined = []
data = []
targets = []

print("Concatenating sitting data.\n")

# Loop exists for each recording and concatenates the data in a similar way to DataShaping.py
# essentially each nth time step is concatenated to the other markers nth time step.
for i in range(0, num_sat_examples):
    if not np.isnan(red_sitting_data[0][i]).any() and not np.isnan(red_sitting_data[1][i]).any():
        red_sitting_data[0][i] = red_sitting_data[0][i].tolist()
        red_sitting_data[1][i] = red_sitting_data[1][i].tolist()
        data_joined.append(red_sitting_data[0][i] + red_sitting_data[1][i])
        targets.append("sitting")

print("Concatenating standing data.\n")

for i in range(0, num_stood_examples):
    if not np.isnan(red_standing_data[0][i]).any() and not np.isnan(red_standing_data[1][i]).any():
        red_standing_data[0][i] = red_standing_data[0][i].tolist()
        red_standing_data[1][i] = red_standing_data[1][i].tolist()
        data_joined.append(red_standing_data[0][i] + red_standing_data[1][i])
        targets.append("standing")

print("concatenating walking data.\n")

for i in range(0, num_walk_examples):
    if not np.isnan(red_walking_data[0][i]).any() and not np.isnan(red_walking_data[1][i]).any():
        red_walking_data[0][i] = red_walking_data[0][i].tolist()
        red_walking_data[1][i] = red_walking_data[1][i].tolist()
        data_joined.append(red_walking_data[0][i] + red_walking_data[1][i])
        targets.append("walking")

# Here the files are stored to be accessed by ClassifierTesting.py
with open("pickle_files/RawDataShaping_results/raw_data_reduced_set", 'wb') as f:
    pickle.dump(data_joined, f)
    f.close()

with open("pickle_files/RawDataShaping_results/raw_data_reduced_set_targets", 'wb') as f:
    pickle.dump(targets, f)
    f.close()

print("RawDataShaping.py complete, please run DataShaping if not already ran,"
      "then move on to ClassifierTesting.py")
