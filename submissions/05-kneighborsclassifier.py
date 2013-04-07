import numpy as np

from sklearn.svm import NuSVR
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree  import DecisionTreeClassifier

def write_out(data):
    f = open('testLabels.csv', 'wb')
    for val in data:
        f.write(str(val) + '\n')

train = np.genfromtxt('../data/train.csv', delimiter=',')
trainLabels = np.genfromtxt('../data/trainLabels.csv', delimiter=',', dtype=int)
test = np.genfromtxt('../data/test.csv', delimiter=',')

clf2 = DecisionTreeClassifier()

clf2 = clf2.fit(train, trainLabels)

validate2 = clf2.predict(train)

print validate2

print sum(abs(validate2 - trainLabels))

#clf2 = LinearSVC(loss = 'l2')

#clf2 = clf2.fit(train[100:200], trainLabels[100:200])

#validate3 = clf2.predict(train[100:200])

#validate4 = np.concatenate((validate2,validate3),axis=0)

#print validate4

#print sum(abs(validate4 - trainLabels[0:200]))

i = 0;
c = 50;


#while c < 1000:
##validate = np.array([])
##i = c;
##print len(train),
##clf2 = LinearSVC(loss = 'l2', fit_intercept = 0)
###clf2 = KNeighborsClassifier()
##for i in range(0,len(train),50):
##    print i
##   
##    #if i == 0:
##    clf2 = clf2.fit(train[i:i+c], trainLabels[i:i+c])
##
##    validate2 = clf2.predict(train[i:i + c])
##
##    validate = np.concatenate((validate,validate2),axis=0)
##
##    print len(validate)
##
##       
##    
##           
##print sum(abs(validate - trainLabels)) 
##    #c += 100;
    
        
    
    
















