import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

iris = sns.load_dataset('iris') 
xiris = iris.drop('species', axis=1)  
yiris = iris['species']

xtrain, xtest, ytrain,ytest = train_test_split(xiris, yiris, random_state = 0)

clf = SVC(kernel='rbf', C=1).fit(xtrain, ytrain)
st.write('Iris dataset')
st.write('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format(clf.score(xtrain, ytrain)))
st.write('Accuracy of RBF SVC classifier on test set: {:.2f}'
     .format(clf.score(xtest, ytest)))

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



from sklearn.metrics import classification_report
from sklearn.svm import SVC
model = SVC()                       
model.fit(xtrain, ytrain)                  
y_model = model.predict(xtest)  

print(classification_report(ytest, y_model))
