from sklearn import svm, metrics
from sklearn.utils import shuffle
import pickle

with open("pickle_files/movement_data", 'rb') as f:
    data = pickle.load(f)

with open("pickle_files/movement_targets", 'rb') as fi:
    target = pickle.load(fi)


print(target[0])
print(*data[0], sep='\n')

n_samples = len(data)

shuffle(data, target)

classifier = svm.SVC(gamma=0.001)

classifier.fit(data[:n_samples // 2], target[:n_samples // 2])

expected = target[n_samples // 2:]

predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
