#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# In[2]:


dataset = pd.read_csv('diabetes.csv')


# In[3]:


dataset


# In[11]:


X = dataset.iloc[:,:-1]


# In[12]:


X


# In[13]:


y = dataset.iloc[:,-1]


# In[14]:


y


# In[15]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state=0)


# In[ ]:





# In[16]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[17]:


regressor.fit(X_train,y_train)


# In[20]:


pickle.dump(regressor,open(r'C:\Users\prash\Desktop\data\diabetes prediction\diabetes.pkl',
                          'wb'))


# In[21]:


model = pickle.load(open(r'C:\Users\prash\Desktop\data\diabetes prediction\diabetes.pkl','rb'))


# In[22]:


print(model.predict([[6,148,72,35,0,33.6,0.627,50]]))


# In[ ]:




