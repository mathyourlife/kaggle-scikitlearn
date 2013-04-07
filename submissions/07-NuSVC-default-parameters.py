import numpy as np
from sklearn.svm import NuSVC
from sklearn import cross_validation
import matplotlib.pyplot as plt


def write_out(data):
    f = open('labels_to_submit.csv', 'wb')
    for val in data:
        f.write(str(val) + '\n')

train = np.genfromtxt('../data/train.csv', delimiter=',')
trainLabels = np.genfromtxt('../data/trainLabels.csv', delimiter=',', dtype=int)
test = np.genfromtxt('../data/test.csv', delimiter=',')

def split_data(training_size=0.5):
    train_data, validation_data, train_labels, validation_labels = cross_validation.train_test_split(train, trainLabels, test_size=training_size)
    return train_data, validation_data, train_labels, validation_labels


def check_fit(predict_labels, labels):
    """
    Check the fit of the model.  Accuracy is the percent of labels
    predicted correctly.
    
    :param predict_labels: fit scikit-learn model
    :type predict_labels: scikit-learn model
    
    :param labels: what the labels should be
    :type labels: numpy ndarray (samples x 1)
    
    :return: percent of labels accurately predicted
    :type: float
    """
    
    accuracy = sum(predict_labels == labels) / float(labels.shape[0])
    #print 'Model accuracy: %0.2f %%' % (accuracy * 100)
    
    return accuracy

def plot_results(plot_x, plot_y):
    """ Simple scatter plot of the results """
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(plot_x, plot_y, '-b')
    ax.grid(True)
    plt.show()

def get_kwargs(val):
    return {
        'degree': val
    }

def testing():
    plot_x = range(1, 10)
    plot_y = []
    for i in xrange(1,10):
        vals = []
        for _ in xrange(20):
            train_data, validation_data, train_labels, validation_labels = split_data()
            clf = NuSVC(**get_kwargs(i))
            clf.fit(train_data, train_labels)
            vals.append(check_fit(clf.predict(validation_data), validation_labels))
        plot_y.append(np.mean(vals))

    plot_results(plot_x, plot_y)

def test_full():
    clf = NuSVC()
    clf.fit(train, trainLabels)
    print check_fit(clf.predict(train), trainLabels)


def final_run():
    clf = NuSVC()
    clf.fit(train, trainLabels)
    predict_labels = clf.predict(test)
    write_out(predict_labels)


if __name__ == '__main__':
    testing()

