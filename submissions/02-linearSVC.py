import numpy as np

train = np.genfromtxt('data/train.csv', delimiter=',')
trainLabels = np.genfromtxt('data/trainLabels.csv', delimiter=',', dtype=int)
test = np.genfromtxt('data/test.csv', delimiter=',')

from sklearn.svm import LinearSVC

clf = LinearSVC(loss = 'l2')

clf = clf.fit(train, trainLabels)

validate = clf.predict(train)

print sum(abs(validate - trainLabels))

testLabels = clf.predict(test)

def write_out(data):
    f = open('testLabels.csv', 'wb')
    for val in data:
        f.write(str(val) + '\n')

