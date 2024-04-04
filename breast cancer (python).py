#!/usr/bin/env python
# coding: utf-8

# In[15]:


#downloading dataset, importing libraries

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


# In[16]:


#reading the dataset, deleting missing value entries
df = pd.read_csv('data-breast cancer.csv')



df.head()

#df.dropna()


# In[17]:


#dropping unnecessary columns to help in selecting features
df.drop(df.columns[[0,4,7,8,9,10,11,12]], axis =1, inplace = True)
df.columns


# In[18]:


#encoding and selecting features
enc = OrdinalEncoder()


df[['diagnosis']] = enc.fit_transform(df[['diagnosis']])


selected_features = df[['radius_mean', 'texture_mean', 'radius_worst', 'texture_worst', 'compactness_worst', 'concave points_worst' ]]
display(selected_features)


# In[19]:


#selecting and scaling X variable (selected features)
scaler = StandardScaler()

X = scaler.fit_transform(selected_features)


# In[20]:


# defining y variable

y = df['diagnosis'].values


# In[21]:


#visualization: box plots of selected features
plt.figure(figsize = (12,5))
sns.boxplot(data = selected_features)


# In[22]:


#visualization: pairplot
plt.figure(figsize = (12,5))
sns.pairplot(data = selected_features)


# In[23]:


#visualization:  heatmap
plt.figure(figsize = (7,8))
sns.heatmap(df)


# In[24]:


#sns.pairplot(df, hue = "diagnosis")


# In[25]:


plt.figure(figsize =(20,8))
sns.countplot(df)


# In[26]:


#visualization: correlation heatmap
plt.figure(figsize =(20,10))
sns.heatmap(df.corr(), annot = True)


# In[27]:


#visualization: violin plots
plt.figure(figsize = (12,5))
sns.violinplot(data= selected_features)


# In[ ]:




