# -*- coding: utf-8 -*-
"""mymall.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TMxbtqifFpDRAN6xWvA8KOgTUG-jwITS
"""

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt

st.header("My Second Streamlit App")
st.write(pd.DataFrame({
    'Intplan': ['yes', 'yes', 'yes', 'no'],
    'Churn Status': [0, 0, 0, 1]
}))

mc = pd.read_csv('/content/mall_customer.csv')

mc.head()

mc.tail()

mc.describe()

mc.info()

x_mc = mc.drop(['Genre','CustomerID'], axis=1)  
x_mc

y_mc = mc['Genre']
y_mc

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(x_mc, y_mc)
Xtrain.head()

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()                       
model.fit(Xtrain, ytrain)                 
y_model = model.predict(Xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest, y_model)

from sklearn.metrics import classification_report
print(classification_report(ytest, y_model))

from sklearn.metrics import confusion_matrix 
confusion_matrix(ytest, y_model)

import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
confusion_matrix = metrics.confusion_matrix(ytest, y_model)

print(confusion_matrix)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=np.unique(y_mc))

cm_display.plot()
plt.show()

from sklearn.metrics import classification_report
print(classification_report(ytest, y_model))
