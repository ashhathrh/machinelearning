import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans

file = "mall_customer.csv"
df= pd.read_csv(file)

features = ['Annual_Income_(k$)', 'Spending_Score']
X = df[features]

a=X.head()
st.write(a)

fig=plt.figure(figsize=(10,4))
plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c='salmon');
st.pyplot(fig)


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
centers

fig=plt.figure(figsize=(10,4))
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
st.pyplot(fig)
