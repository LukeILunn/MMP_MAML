from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle
import pickle
import numpy as np

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

# print(target[0])
# print(*data[0], sep='\n')

cross_val_scores = []
cross_val_avgs = []
cross_val_stdev = []

train_frac = 0.7
vald_frac = 0.2
max_its = 5500
layer_sizes = 600

n_samples = int(len(data) * train_frac)
r_n_samples = int(len(red_data) * train_frac)
r_r_n_samples = int(len(red_raw_data) * train_frac)

data = np.asarray(data, dtype=np.float64)
target = np.asarray(target)
red_data = np.asarray(red_data, dtype=np.float64)
red_target = np.asarray(red_target)
red_raw_data = np.asarray(red_raw_data, dtype=np.float64)
red_raw_targets = np.asarray(red_raw_targets)


data = data.reshape(len(data), -1)
red_raw_data = red_raw_data.reshape(len(red_raw_data), -1)

classifier_for_red_raw = MLPClassifier(hidden_layer_sizes=(layer_sizes), max_iter=max_its, validation_fraction=vald_frac, shuffle=False)
classifier_for_red = MLPClassifier(hidden_layer_sizes=(layer_sizes), max_iter=max_its, validation_fraction=vald_frac, shuffle=False)
classifier = MLPClassifier(hidden_layer_sizes=(layer_sizes), max_iter=max_its, validation_fraction=vald_frac, shuffle=False)


# while layer_sizes <= 600:
# classifier_for_red = MLPClassifier(hidden_layer_sizes=(layer_sizes), max_iter=max_its,
#                                    validation_fraction=vald_frac, shuffle=False)
# classifier = MLPClassifier(hidden_layer_sizes=(layer_sizes), max_iter=max_its, validation_fraction=vald_frac,
#                            shuffle=False)

# print("\n", layer_sizes, "\n")
# cross_val_avgs = []
# cross_val_stdev = []

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

# with open("pickle_files/final_comparison/cross_val_scores", 'wb') as f:
#     pickle.dump(cross_val_scores, f)
#     f.close()

with open("pickle_files/final_comparison/cross_val_avgs", 'wb') as f:
    pickle.dump(cross_val_avgs, f)
    f.close()

with open("pickle_files/cross_val_stdev", 'wb') as f:
    pickle.dump(cross_val_stdev, f)
    f.close()

# with open("pickle_files/reduced_marker_tests/cross_val_scores_lw_sob", 'wb') as f:
#     pickle.dump(cross_val_scores, f)
#     f.close()
#
# with open("pickle_files/reduced_marker_tests/cross_val_avgs_lw_sob", 'wb') as f:
#     pickle.dump(cross_val_avgs, f)
#     f.close()
#
# with open("pickle_files/reduced_marker_tests/cross_val_stdev_lw_sob", 'wb') as f:
#     pickle.dump(cross_val_stdev, f)
#     f.close()

# with open("pickle_files/reduced_marker_tests/cross_val_scores_25", 'wb') as f:
#     pickle.dump(cross_val_scores, f)
#     f.close()
#
# with open("pickle_files/reduced_marker_tests/cross_val_avgs_25", 'wb') as f:
#     pickle.dump(cross_val_avgs, f)
#     f.close()
#
# with open("pickle_files/reduced_marker_tests/cross_val_stdev_25", 'wb') as f:
#     pickle.dump(cross_val_stdev, f)
#     f.close()

# with open("pickle_files/single_layer_tests/cross_val_avgs_layer_size_"+str(layer_sizes), 'wb') as f:
#     pickle.dump(cross_val_avgs, f)
#     f.close()
#
# with open("pickle_files/single_layer_tests/cross_val_stdev_layer_size_"+str(layer_sizes), 'wb') as f:
#     pickle.dump(cross_val_stdev, f)
#     f.close()

# with open("pickle_files/max_iterations_tests/cross_val_avgs_its_" + str(max_its), 'wb') as f:
#     pickle.dump(cross_val_avgs, f)
#     f.close()
#
# with open("pickle_files/max_iterations_tests/cross_val_stdev_its_" + str(max_its), 'wb') as f:
#     pickle.dump(cross_val_stdev, f)
#     f.close()

# layer_sizes += 50

data, target = shuffle(data, target)
red_data, red_target = shuffle(red_data, red_target)
red_raw_data, red_raw_targets = shuffle(red_raw_data, red_raw_targets)

classifier.fit(data[:n_samples], target[:n_samples])
classifier_for_red.fit(red_data[:r_n_samples], red_target[:r_n_samples])
classifier_for_red_raw.fit(red_raw_data[:r_r_n_samples], red_raw_targets[:r_r_n_samples])

red_raw_expected = red_raw_targets[r_r_n_samples:]
red_expected = red_target[r_n_samples:]
expected = target[n_samples:]

red_raw_predicted = classifier_for_red_raw.predict(red_raw_data[r_r_n_samples:])
red_predicted = classifier_for_red.predict(red_data[r_n_samples:])
predicted = classifier.predict(data[n_samples:])

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
