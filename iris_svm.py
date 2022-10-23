import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris= load_iris()
(xiris, yiris) = load_iris(return_X_y = True)
xtrain, xtest, ytrain,ytest = train_test_split(xiris, yiris, random_state = 0)


clf = SVC(kernel='rbf', C=1).fit(xtrain, ytrain)
print('Iris dataset')
print('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format(clf.score(xtrain, ytrain)))
print('Accuracy of RBF SVC classifier on test set: {:.2f}'
     .format(clf.score(xtest, ytest)))

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()                       
model.fit(xtrain, ytrain)                  
y_model = model.predict(xtest)  

print(classification_report(ytest, y_model))

# Confusion Matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix(ytest, y_model)

from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
svm = SVC(random_state=42, kernel='linear')

# Fit the data to the SVM classifier
svm = svm.fit(xtrain, ytrain)

# Evaluate by means of a confusion matrix
matrix = plot_confusion_matrix(svm, xtest, ytest, cmap=plt.cm.Blues, normalize='true')
plt.title('Confusion matrix for linear SVM')
plt.show(matrix)
plt.show()
