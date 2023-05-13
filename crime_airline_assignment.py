#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[12]:


data=pd.read_csv(r"C:\Users\sagar\Desktop\sagar\sagar_assignment\Assignment7\EastWestAirlines1.csv")


# In[13]:


data.shape


# In[14]:


data.head()


# In[15]:


data.isna().sum()


# In[16]:


data.info()


# In[17]:


data.head(2)


# In[18]:


def norm_fun(i):
    x=((i-i.min())/(i.max()-i.min()))
    return(x)


# In[19]:


df_norm=norm_fun(data.iloc[:,1:])
df_norm


# In[20]:


dendrogram=sch.dendrogram(sch.linkage(df_norm,method='complete'))


# In[21]:


hc=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='complete')


# In[22]:


y_hc=hc.fit_predict(df_norm)


# In[23]:


cluster=pd.DataFrame(y_hc,columns=['clusters'])
cluster.value_counts()


# In[24]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[25]:


df_norm.head()


# In[26]:


model=KMeans()
model.fit(df_norm)
model.labels_


# In[27]:


md=pd.Series(model.labels_)
data['clust']=md
df_norm.head()


# In[28]:


data.groupby(data.clust).mean()


# In[29]:


data.head()


# In[30]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[31]:


array=data.values
array


# In[32]:


stscaler=StandardScaler().fit(array)
x=stscaler.transform(array)


# In[33]:


dbscan=DBSCAN(eps=0.8,min_samples=6)
dbscan.fit(x)
dbscan.labels_


# In[34]:


cl=pd.DataFrame(dbscan.labels_,columns=['Cluster'])
cl


# In[ ]:




