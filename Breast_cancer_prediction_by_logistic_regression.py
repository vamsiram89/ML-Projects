#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[12]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[13]:


df = pd.read_csv("cancer_pred_Data.csv")
df.head()


# ## Information of data

# In[14]:


df.info()


# # know the descriptive statistics

# In[15]:


df.describe()


# In[16]:


sns.heatmap(df.isnull())


# # finding the null values 

# In[17]:


df.isnull().sum()


# ## Drop the unnecessary columns

# In[18]:


df.drop(["Unnamed: 32","id"],axis = 1,inplace = True)


# In[19]:


df.head()


# # converting data into o and 1

# In[20]:


df.diagnosis = [1 if value == "M" else 0 for value in df.diagnosis]


# In[22]:


df.head()


# In[29]:


df["diagnosis"].value_counts().plot(kind = "bar",color = ["Red","Green"])


# ## Divide Target variable and predictors

# In[38]:


y = df["diagnosis"]
x = df.drop(["diagnosis"],axis = 1)


# # Normalisation

# In[43]:


from sklearn.preprocessing import StandardScaler
x_scaled = StandardScaler().fit_transform(x)
x_scaled


# ## Spliting the Data

# In[46]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size = 0.3,random_state = 42)


# # Train the Model

# In[ ]:


from sklearn.linear_model import LogisticRegression

# create the lr model
lr = LogisticRegression()

# train the model on training data
lr.fit(x_train,y_train)

# Predict the target variable on test data
y_pred = lr.predict(x_test)


# ### Output for target variable 

# In[59]:


y_pred


# In[60]:


y_test


# # Evaluation of the model

# In[65]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred,y_test)
print(accuracy)


# In[66]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:




