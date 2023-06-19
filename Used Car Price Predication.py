#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[3]:


# loading the data from csv file to pandas dataframe
car_data =pd.read_csv('car data.csv')


# In[4]:


#checking the first 5 rows of dataset
car_data.head()


# In[5]:


# checking the number of rows and columns
car_data.shape


# In[6]:


# getting some information about the dataset
car_data.info()


# In[7]:


# checking the number of missing values
car_data.isnull().sum()


# In[8]:


#There is no missing values in the data set.


# In[9]:


#check for catagorical data 
car_data.Car_Name.value_counts()


# In[10]:


car_data.Fuel_Type.value_counts()


# In[11]:


car_data.Seller_Type.value_counts()


# In[12]:


car_data.Transmission.value_counts()


# In[13]:


# encoding "Fuel_Type" Column
car_data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)


# In[14]:


# encoding "Seller_Type" Column
car_data.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)


# In[16]:


# encoding "Transmission" Column
car_data.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[17]:


car_data.head()


# In[19]:


car_data.tail(10)


# In[44]:


#spillting the data into independent and target variables
X = car_data.drop(["Car_Name",'Selling_Price'],axis=1)
Y = car_data['Selling_Price']


# In[45]:


X


# In[46]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)


# In[47]:


X_train


# In[48]:


X_test


# In[49]:


X_test.nunique()


# In[50]:


from sklearn.linear_model import LinearRegression

#creating linear regression model
linear_model = LinearRegression()


# In[51]:


linear_model.fit(X_train,Y_train)


# In[53]:


linear_model.score(X_train,Y_train)


# In[54]:


y_train_pred=linear_model.predict(X_train)


# In[55]:


print(y_train_pred)


# In[57]:


res = Y_train - y_train_pred
res


# In[58]:


plt.figure(figsize=(15,8))
sns.distplot(res)


# In[59]:


from sklearn.metrics import r2_score

r2_score(Y_train, y_train_pred)


# In[60]:


# make predictions using xtest

y_test_pred = linear_model.predict(X_test)


# In[61]:


# evaluating the model

from sklearn.metrics import r2_score

r2_score(Y_test, y_test_pred)


# In[62]:


# checking the error in the model

from sklearn.metrics import mean_absolute_error

mean_absolute_error(Y_test, y_test_pred)


# In[ ]:


2.  LASSO REGRESSION MODEL


# In[63]:


# loading the linear regression model
lass_reg_model = Lasso()


# In[64]:


lass_reg_model.fit(X_train,Y_train)


# In[ ]:


# MODEL EVALUTION


# In[65]:


# prediction on Training data
training_data_prediction = lass_reg_model.predict(X_train)


# In[66]:


# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)


# In[67]:


# prediction on Test data
test_data_prediction = lass_reg_model.predict(X_test)


# In[68]:


# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)


# In[ ]:




