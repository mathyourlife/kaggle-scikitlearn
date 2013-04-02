import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
import matplotlib.pyplot as plt

def write_out(data):
    f = open('testLabels.csv', 'wb')
    for val in data:
        f.write(str(val) + '\n')

train = np.genfromtxt('../data/train.csv', delimiter=',')
trainLabels = np.genfromtxt('../data/trainLabels.csv', delimiter=',', dtype=int)
test = np.genfromtxt('../data/test.csv', delimiter=',')

def model_parameters(model_params=None):
    """
    From the documentation on KNeighborsClassifier.  

    parameter: default (other options)

    n_neighbors: 5
    weights: 'uniform' ('distance', user defined function)
    algorithm: 'auto' ('ball_tree', 'kd_tree', 'brute')
    leaf_size: 30
    warn_on_equidistant: True
    p: 2
    """

    if model_params is None:
        model_params = {}
    
    static_values = {
        'n_neighbors': 7, # 7 seem to peak at just less than 0.9 
        'weights': 'distance',
        'algorithm': 'auto',
        'leaf_size': 30,
        'warn_on_equidistant': True,
        'p': 2
    }

    static_values.update(model_params)

    return static_values


def define_model(model_params):

    params = model_parameters(model_params)
    
    #print 'Creating KNeighbors Classifier with parameters'

    model = KNeighborsClassifier(**params)
    
    #print model
    return model

def fit_model(clf, data, labels):
    """
    Run the supervised learning method on the supplied model
    with the data set and associated labels.
    
    :param clf: Classification model
    :type clf: scikit-learn model to be fit
    
    :param data: data set that will be used to predict the label
    :type data: numpy ndarray (samples x properties)
    
    :param labels: The set of label that correspond to the 
    :type labels: numpy ndarray (samples x 1)
    
    :return: Fitted model
    :rtype: type(clf)
    """
    
    fit_model = clf.fit(data, labels)

    return fit_model

def check_fit(clf, data, labels):
    """
    Check the fit of the model.  Accuracy is the percent of labels
    predicted correctly.
    
    :param clf: fit scikit-learn model
    :type clf: scikit-learn model
    
    :param data: array of data to feed into the model
    :type data: numpy ndarray (samples x properties)
    
    :param labels: what the labels should be
    :type labels: numpy ndarray (samples x 1)
    
    :return: percent of labels accurately predicted
    :type: float
    """
    predict_labels = clf.predict(data)
    
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

def test_parameters():
    plot_x = range(1, 20)
    plot_y = []
    for i in plot_x:
        #print '=' * 100
        
        vals = []
        for _ in xrange(200):
            train_data, validation_data, train_labels, validation_labels = cross_validation.train_test_split(train, trainLabels, test_size=0.25)
            
            set_params = {
                'n_neighbors': i,
            }
            params = model_parameters(set_params)
            clf = define_model(params)

            clf = fit_model(clf, train_data, train_labels)
            val = check_fit(clf, validation_data, validation_labels)

            vals.append(val)

        plot_y.append(np.mean(vals))

    plot_results(plot_x, plot_y)


def test_labels():
    params = model_parameters()
    clf = define_model(params)

    clf = fit_model(clf, train, trainLabels)
    predict_labels = clf.predict(test)
    write_out(predict_labels)


def main():
    test_labels()


if __name__ == '__main__':
    main()
