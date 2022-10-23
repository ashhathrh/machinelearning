import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

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

#classification report
model = SVC()                       
model.fit(xtrain, ytrain)                  
y_model = model.predict(xtest)  

cr=classification_report(ytest, y_model)
st.write(cr)

confusion_matrix(yest,y_model)

#confusion matrix
confusion_matrix=metrics.confusion_matrix(ytest,y_model)
c=confusion_matrix
st.write("confusion matrix:",c)
fig=plt.figure(figsize=(10,4))
sns.heatmap(confusion_matrix,annot=True)
st.pyplot(fig)





