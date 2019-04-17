from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.utils import shuffle
import pickle
import numpy as np

with open("pickle_files/movement_data", 'rb') as f:
    data = pickle.load(f)
    f.close()

with open("pickle_files/movement_targets", 'rb') as fi:
    target = pickle.load(fi)
    fi.close()

with open("pickle_files/red_movement_data", 'rb') as f:
    red_data = pickle.load(f)
    f.close()

with open("pickle_files/red_movement_targets", 'rb') as fi:
    red_target = pickle.load(fi)
    fi.close()

with open("pickle_files/raw_data_reduced_set", 'rb') as f:
    red_raw_data = pickle.load(f)
    f.close()

with open("pickle_files/raw_data_reduced_set_targets", 'rb') as fi:
    red_raw_targets = pickle.load(fi)
    fi.close()

# print(target[0])
# print(*data[0], sep='\n')

n_samples = len(data)
r_n_samples = len(red_data)
r_r_n_samples = len(red_raw_data)

data = np.asarray(data, dtype=np.float64)
red_data = np.asarray(red_data, dtype=np.float64)
red_raw_data = np.asarray(red_raw_data, dtype=np.float64)

data, target = shuffle(data, target)
red_data, red_target = shuffle(red_data, red_target)
red_raw_data, red_raw_targets = shuffle(red_raw_data, red_raw_targets)

data = data.reshape(n_samples, -1)
red_raw_data = red_raw_data.reshape(r_r_n_samples, -1)

classifier_for_red_raw = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, validation_fraction=0.1)
classifier_for_red = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, validation_fraction=0.1)
classifier = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, validation_fraction=0.1)

classifier.fit(data[:n_samples // 2], target[:n_samples // 2])
classifier_for_red.fit(red_data[:r_n_samples // 2], red_target[:r_n_samples // 2])
classifier_for_red_raw.fit(red_raw_data[:r_r_n_samples // 2], red_raw_targets[:r_r_n_samples // 2])

red_raw_expected = red_raw_targets[r_r_n_samples // 2:]
red_expected = red_target[r_n_samples // 2:]
expected = target[n_samples // 2:]

red_raw_predicted = classifier_for_red_raw.predict(red_raw_data[r_r_n_samples // 2:])
red_predicted = classifier_for_red.predict(red_data[r_n_samples // 2:])
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print("\n\n\n")

print("Classification report for classifier %s:\n%s\n"
      % (classifier_for_red, metrics.classification_report(red_expected, red_predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(red_expected, red_predicted))

print("\n\n\n")

print("Classification report for classifier %s:\n%s\n"
      % (classifier_for_red_raw, metrics.classification_report(red_raw_expected, red_raw_predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(red_raw_expected, red_raw_predicted))
