#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


# In[2]:


data=pd.read_csv(r"C:\Users\sagar\Desktop\sagar\sagar_assignment\Assignment7\crime_data.csv")
data.head()


# In[3]:


data.isna().sum()


# In[4]:


data[data.duplicated()]


# In[5]:


def norm_func(i):
    x=(i-i.min()/i.max()-i.min())
    return(x)


# In[6]:


df_norm=norm_func(data.iloc[:,1:])
df_norm


# In[7]:


hc=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='single')


# In[8]:


y_hc=hc.fit_predict(df_norm)
y_hc


# In[9]:



clusters=pd.DataFrame(y_hc,columns=['Clusters'])
clusters


# In[10]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[11]:


model=KMeans(n_clusters=4)
model.fit(df_norm)
model.labels_


# In[12]:


md=pd.Series(model.labels_)
data['clust']=md
df_norm.head()


# In[13]:


data.iloc[:,1:7].groupby(data.clust).mean()


# In[14]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[16]:


crime=pd.read_csv(r"C:\Users\sagar\Desktop\sagar\sagar_assignment\Assignment7\crime_data.csv")
crime.head()


# In[17]:


crime1=crime.iloc[:,1:6]
crime1.head()


# In[18]:


array=crime1.values
array


# In[19]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array)


# In[20]:


dbscan=DBSCAN(eps=0.8,min_samples=5)


# In[21]:


dbscan.fit(X)


# In[22]:


cl=pd.DataFrame(dbscan.labels_,columns=["cluster"])
cl


# In[ ]:




