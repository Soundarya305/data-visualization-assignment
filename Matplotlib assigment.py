#!/usr/bin/env python
# coding: utf-8

# In[3]:


##In this assignment students have to transform iris data into 3 dimensionsand plot a 3d chart with transformed dimensions and colour each datapoint with specific class.

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import datasets


# In[4]:


iris = datasets.load_iris()


# In[5]:


X_reduced = PCA(n_components=3).fit_transform(iris.data)
Y = iris.target

fig = plt.figure(1, figsize=(8,6))
ax = Axes3D(fig, elev=-150, azim=110)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y, cmap=plt.cm.rainbow_r)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")

plt.show()


# In[ ]:




