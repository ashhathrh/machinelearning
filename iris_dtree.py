import seaborn as sns
import pandas as pd
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

iris = sns.load_dataset('iris')

xiris = iris.drop('species', axis=1)  
yiris = iris['species']

xtrain, xtest, ytrain, ytest = train_test_split(xiris, yiris,random_state=1)

from sklearn.tree import DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(xtrain, ytrain)


clf.fit(xtrain, ytrain)

tree.plot_tree(clf.fit(xtrain, ytrain) )

clf.score(xtest, ytest)
