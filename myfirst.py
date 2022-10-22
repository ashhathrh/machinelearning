import seaborn as sns
import pandas as pd
iris = sns.load_dataset('iris')
sns.boxplot(data=iris)
