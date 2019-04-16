from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.utils import shuffle
import pickle
import numpy as np

with open("pickle_files/movement_data", 'rb') as f:
    data = pickle.load(f)

with open("pickle_files/movement_targets", 'rb') as fi:
    target = pickle.load(fi)


print(target[0])
print(*data[0], sep='\n')

n_samples = len(data)
data = np.asarray(data, dtype=np.float64)

classifier = MLPClassifier()

classifier.fit(data[:n_samples // 2], target[:n_samples // 2])

expected = target[n_samples // 2:]

predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
