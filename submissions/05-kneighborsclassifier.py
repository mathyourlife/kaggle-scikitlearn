import numpy as np

def write_out(data):
    f = open('testLabels.csv', 'wb')
    for val in data:
        f.write(str(val) + '\n')

train = np.genfromtxt('../data/train.csv', delimiter=',')
trainLabels = np.genfromtxt('../data/trainLabels.csv', delimiter=',', dtype=int)
test = np.genfromtxt('../data/test.csv', delimiter=',')

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf = clf.fit(train, trainLabels)

validate = clf.predict(train)

print sum(abs(validate - trainLabels))

testLabels = clf.predict(test)

from sklearn.svm import LinearSVC

clf2 = LinearSVC(loss = 'l2')

clf2 = clf2.fit(train, trainLabels)

validate2 = clf2.predict(train)

print sum(abs(validate2 - trainLabels))

testLabels = clf2.predict(test)

from sklearn.neighbors import  KNeighborsClassifier

clf3 = KNeighborsClassifier()

clf3 = clf3.fit(train, trainLabels)

validate3 = clf3.predict(train)

print validate3

print sum(abs(validate3 - trainLabels))

testLabels = clf3.predict(test)

write_out (testLabels)


retry =  train[validate3 != trainLabels]
retrylabel =  trainLabels[validate3 != trainLabels]

print retry

clf2 = LinearSVC(loss = 'l2')

clf2 = clf2.fit(retry, retrylabel)

validate2 = clf2.predict(retry)

print sum(abs(validate2 - retrylabel))










