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
st.write("  KMeans Clustering for Mall Customers")
st.write(a)

st.write("  Scatter plot for annual income and spending score dataset")
fig=plt.figure(figsize=(10,4))
plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c='red');
st.pyplot(fig)


kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

st.write("  Scatter plot with colours for annual income and spending score dataset")

centers = kmeans.cluster_centers_
centers


fig=plt.figure(figsize=(10,4))
plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
st.pyplot(fig)
