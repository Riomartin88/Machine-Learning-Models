#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df=pd.read_csv('Salary_Data.csv')


# In[4]:


df.head()


# In[6]:


df.describe


# In[10]:


from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[11]:


#inbuilt function in sl;earn for bui;lting the dataset
#100 columns with 2 i/p and 1 o/p
x,y=make_regression(n_samples=100,n_features=2,n_informative=2,n_targets=1,noise=50)


# In[13]:


#converting into dataframe 
df=pd.DataFrame({'feature1':x[:,0],'feature2':x[:,1],'target':y})


# In[14]:


df.head()


# In[16]:


#data visualisation of above multiple linear regression dataset
fig=px.scatter_3d(df,x='feature1',y='feature2',z='target')
fig.show()


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)


# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


lr=LinearRegression()


# In[20]:


lr.fit(x_train,y_train)


# In[21]:


y_pred=lr.predict(x_test)


# In[26]:


mse=mean_squared_error(y_test,y_pred)
mse


# In[27]:


mae=mean_absolute_error(y_test,y_pred)
mae


# In[30]:


#


# In[40]:


x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
xGrid, yGrid = np.meshgrid(y, x)


final = np.vstack((xGrid.ravel().reshape(1,100),yGrid.ravel().reshape(1,100))).T
z_final = lr.predict(final).reshape(10,10)

z = z_final


# In[41]:


fig = px.scatter_3d(df, x='feature1', y='feature2', z='target')

fig.add_trace(go.Surface(x = x, y = y, z =z ))

fig.show()


# In[42]:


lr.coef_


# In[43]:


lr.intercept_


# In[ ]:




