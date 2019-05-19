from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.utils import shuffle
import pickle
import numpy as np


# Here all the variables are created containing the results of both DataShaping.py and RawDataShaping.py
with open("pickle_files/DataShaping_results/movement_data", 'rb') as f:
    data = pickle.load(f)
    f.close()

with open("pickle_files/DataShaping_results/movement_targets", 'rb') as fi:
    target = pickle.load(fi)
    fi.close()

with open("pickle_files/DataShaping_results/red_movement_data", 'rb') as f:
    red_data = pickle.load(f)
    f.close()

with open("pickle_files/DataShaping_results/red_movement_targets", 'rb') as fi:
    red_target = pickle.load(fi)
    fi.close()

with open("pickle_files/RawDataShaping_results/raw_data_reduced_set", 'rb') as f:
    red_raw_data = pickle.load(f)
    f.close()

with open("pickle_files/RawDataShaping_results/raw_data_reduced_set_targets", 'rb') as fi:
    red_raw_targets = pickle.load(fi)
    fi.close()

# These empty lists are used to store the results of the k-fold cross validation classification.
cross_val_scores = []
cross_val_avgs = []
cross_val_stdev = []

# Parameters of the MLPClassifier can be changed here to experiment with the effect of parameters
train_frac = 0.7
vald_frac = 0.2
max_its = 5500
layer_sizes = 600

# Creating the size of the training fraction for the random shuffle classification.
n_samples = int(len(data) * train_frac)
r_n_samples = int(len(red_data) * train_frac)
r_r_n_samples = int(len(red_raw_data) * train_frac)

# All stored variables become NumPy arrays to allow for processes such as flatten() or reshape().
data = np.asarray(data, dtype=np.float64)
target = np.asarray(target)
red_data = np.asarray(red_data, dtype=np.float64)
red_target = np.asarray(red_target)
red_raw_data = np.asarray(red_raw_data, dtype=np.float64)
red_raw_targets = np.asarray(red_raw_targets)

# It is necessary for the data to be reshaped in order for the dimensions to be accepted by the classifier.
data = data.reshape(len(data), -1)
red_raw_data = red_raw_data.reshape(len(red_raw_data), -1)

# Here a separate classifier is created for each of the dataset types, full marker set with features, reduced
# marker set with features, and the reduced marker set with raw data.
classifier_for_red_raw = MLPClassifier(hidden_layer_sizes=(layer_sizes), max_iter=max_its, validation_fraction=vald_frac, shuffle=False)
classifier_for_red = MLPClassifier(hidden_layer_sizes=(layer_sizes), max_iter=max_its, validation_fraction=vald_frac, shuffle=False)
classifier = MLPClassifier(hidden_layer_sizes=(layer_sizes), max_iter=max_its, validation_fraction=vald_frac, shuffle=False)

# The commented while loop here was used when experimenting with various sizes of layers and various
# max iteration values. This way the program could be left to run and would create pickle files for
# each of these values. As it can take a very long time to test a three-layer neural network on this
# data this was necessary to make the process quicker.

# while layer_sizes <= 600:
# classifier_for_red = MLPClassifier(hidden_layer_sizes=(layer_sizes), max_iter=max_its,
#                                    validation_fraction=vald_frac, shuffle=False)
# classifier = MLPClassifier(hidden_layer_sizes=(layer_sizes), max_iter=max_its, validation_fraction=vald_frac,
#                            shuffle=False)
# print("\n", layer_sizes, "\n")
# cross_val_avgs = []
# cross_val_stdev = []

# For the purposes of demonstration of the program this for loop will run 30 tests of 5-fold cross validation
# storing the scores in pickle_files/final_comparison. As it is, the files needed for the final
# comparison will be created and then Visualise.py can be run to see the results.
for i in range(0, 30):

    print("\n", i, "\n")
    store_avg = []
    store_stdev = []

    scores = cross_val_score(classifier, data, target, cv=5)
    print('\n', scores)
    print("\nAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

    store_avg.append(scores.mean())
    store_stdev.append(scores.std())

    red_scores = cross_val_score(classifier_for_red, red_data, red_target, cv=5)
    print('\n', red_scores)
    print("\nAccuracy: %0.2f (+/- %0.2f)" % (red_scores.mean(), red_scores.std()))

    store_avg.append(red_scores.mean())
    store_stdev.append(red_scores.std())

    red_raw_scores = cross_val_score(classifier_for_red_raw, red_raw_data, red_raw_targets, cv=5)
    print('\n', red_raw_scores)
    print("\nAccuracy: %0.2f (+/- %0.2f)\n" % (red_raw_scores.mean(), red_raw_scores.std()))

    store_avg.append(red_raw_scores.mean())
    store_stdev.append(red_raw_scores.std())

    cross_val_avgs.append(store_avg)
    cross_val_stdev.append(store_stdev)

# Depending upon the current experiment the pickle file would have to store information to different files
# Editing these lines will change where the information is stored and the results of the many experiments are
# stored within the pickle_files folder for viewing.
with open("pickle_files/final_comparison/cross_val_avgs", 'wb') as f:
    pickle.dump(cross_val_avgs, f)
    f.close()

with open("pickle_files/final_comparison/cross_val_stdev", 'wb') as f:
    pickle.dump(cross_val_stdev, f)
    f.close()

# layer_sizes += 50

# Shuffling the datasets, ensures that the training subset is a random combination of the data
# but also ensures that the target will still correspond to the correct time step.
data, target = shuffle(data, target)
red_data, red_target = shuffle(red_data, red_target)
red_raw_data, red_raw_targets = shuffle(red_raw_data, red_raw_targets)

# Here the classifiers are trained using fit() upon the subset of the data which was defined at the top of
# this script.
classifier.fit(data[:n_samples], target[:n_samples])
classifier_for_red.fit(red_data[:r_n_samples], red_target[:r_n_samples])
classifier_for_red_raw.fit(red_raw_data[:r_r_n_samples], red_raw_targets[:r_r_n_samples])

# Storing the ground truths to allow for the creation of the classification report and confusion matrix.
red_raw_expected = red_raw_targets[r_r_n_samples:]
red_expected = red_target[r_n_samples:]
expected = target[n_samples:]

# Test the classifiers upon the remaining data which wasn't used for training.
red_raw_predicted = classifier_for_red_raw.predict(red_raw_data[r_r_n_samples:])
red_predicted = classifier_for_red.predict(red_data[r_n_samples:])
predicted = classifier.predict(data[n_samples:])

# These print statements produce the classification report and confusion matrix for the test which was run using
# shuffle(), fit(), and predict().
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print("\n")

print("Classification report for classifier %s:\n%s\n"
      % (classifier_for_red, metrics.classification_report(red_expected, red_predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(red_expected, red_predicted))

print("\n")

print("Classification report for classifier %s:\n%s\n"
      % (classifier_for_red_raw, metrics.classification_report(red_raw_expected, red_raw_predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(red_raw_expected, red_raw_predicted))

print("\n ClassifierTesting.py is now complete, run Visualise.py to view the results of the tests.")
