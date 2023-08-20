#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Created on Sun Aug 20 23:46:28 2023

# @author: atm
# """

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
from sklearn.feature_selection import VarianceThreshold


# In[2]:


def load_data(file_path, file_format):
    if file_format == 'csv':
        return pd.read_csv(file_path)
    elif file_format == 'excel':
        return pd.read_excel(file_path)
    elif file_format == 'sql':
        engine = create_engine('sqlite:///your_database.db')  # Replace with your database connection
        query = 'SELECT * FROM your_table'
        return pd.read_sql(query, engine)
    
def handle_missing_values(data):
    return data.dropna()


# In[3]:


def encode_categorical_features(data):
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        data[col] = LabelEncoder().fit_transform(data[col])
    return data


# In[4]:


def scale_numerical_features(data):
    num_cols = data.select_dtypes(include=['float64']).columns
    data[num_cols] = StandardScaler().fit_transform(data[num_cols])
    return data


# In[5]:


def feature_selection(data):
    X = data
    selector = VarianceThreshold(threshold=0.01)  # Adjust the threshold as needed
    selected_features = selector.fit_transform(X)
    selected_indices = selector.get_support(indices=True)
    selected_column_names = X.columns[selected_indices]
    return data[selected_column_names]


# In[6]:


def dimensionality_reduction(data):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    return reduced_data


# In[7]:


def visualize_numeric_columns(data):
    sns.set()
    num_cols = data.select_dtypes(include=['float64']).columns
    for col in num_cols:
        plt.figure()
        sns.histplot(data[col], bins=20, kde=True)
        plt.title(f'Histogram of {col}')
        plt.show()


# In[8]:


def visualize_categorical_columns(data):
    sns.set()
    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        plt.figure()
        sns.countplot(data=data, x=col)
        plt.title(f'Count Plot of {col}')
        plt.xticks(rotation=45)
        plt.show()


# In[9]:


def visualize_interactive(data):
    reduced_data = dimensionality_reduction(data)
    fig = px.scatter(reduced_data, x=0, y=1)
    fig.show()


# In[ ]:


def automated_eda(file_path, file_format):
    data = load_data(file_path, file_format)
    data = handle_missing_values(data)
    data = encode_categorical_features(data)
    data = scale_numerical_features(data)
    data = feature_selection(data)
    visualize_numeric_columns(data)
    visualize_categorical_columns(data)
    visualize_interactive(data)


# In[ ]:


def main():
    file_path = input("Enter the file path: ")
    file_format = input("Enter the file format (csv/excel/sql): ")
    automated_eda(file_path, file_format)
if __name__ == "__main__":
    main()

