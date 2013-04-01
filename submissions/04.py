import numpy as np

train = np.genfromtxt('data/train.csv', delimiter=',')
trainLabels = np.genfromtxt('data/trainLabels.csv', delimiter=',', dtype=int)
test = np.genfromtxt('data/test.csv', delimiter=',')


X = np.matrix(train)
y = np.matrix(trainLabels)

beta = (X.T * X).I * X.T * y.T
print beta.T

test_matrix = np.matrix(test)

guess = test_matrix * beta
test_y = np.round(guess)

test_y[test_y > 1] = 1
test_y[test_y < 0] = 0

def write_matrix(matrix):
    f = open('testLabels.csv', 'wb')
    for val in matrix:
        f.write(str(int(val[0,0])) + '\n')

check_train = X * beta
check_train = np.round(check_train)
check_train[check_train > 1] = 1
check_train[check_train < 0] = 0

sum(abs(check_train - y.T))

import matplotlib.pyplot as plt

col = 15
plt.figure(1)

for i in np.arange(4):
    idx = i + 1
    ax = plt.subplot(4, 2, (2*idx) - 1)
    plt.hist(train[trainLabels==0, col])
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    ax = plt.subplot(4, 2, (2*idx))
    plt.hist(train[trainLabels==1, col])
    ax.set_yticklabels([])
    ax.set_xticklabels([])

plt.show()
