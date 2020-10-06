#!/usr/bin/env python
# coding: utf-8

# In[47]:


#IMPORTING ALL THE NECESSARY LIBRARIES
import pandas as pd
import numpy as np
import os
import math
import matplotlib


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


os.getcwd()


# In[5]:


os.chdir("D:\\DailyuPython")


# Reading the csv file into data frame "titanic"

# In[6]:


titanic=pd.read_csv("train.csv")


# how our dataset looks like??

# In[7]:


titanic


# In[ ]:





# #Doing expolratory data analysis
# Reading of data we have 12 columns 
# 
# 1 column represents passengers ID which starts from number 1
# 2 column is a categoric variable which have 0 and 1 as outputs
# 3 column represents the passengers travelling in which class whether first or second or third class labelled as 1 2 and 3
# 4 column represents the names of passengers
# 5 and 6 columns represnts sex and age respectively
# 7 column represents the number of siblings or spouse who were travelling together
# 8 column parch means number of parents or children aboarding titanic
# 9 represents the ticket number 
# 10 represents the price of the ticket
# 11 represents cabin
# 12 represents embarked with 3 uiquevalues
# 
# 

# Analyzing data
#  
#  By using the seaborn library(sns) we are plotting the survival passengers 
# 

# In[8]:


sns.countplot(x='Survived',data=titanic)


# Knowing gender affect on survival data

# In[9]:


sns.countplot(x="Survived",hue="Sex",data=titanic)


# In[10]:


sns.countplot(x="Survived",hue="Pclass",data=titanic)


# In[11]:


sns.countplot(x="SibSp",data=titanic)


# In[12]:


sns.countplot(x="Parch",data=titanic)


# As part of Data Wrangling
# Let's know about the Null or missing values in the titanic dataset

# In[13]:


titanic.isnull().sum()


# Heat map does provide us with a better visualization WRT null values 
# As seen in heat map null values are present in 3 columns
# 1.Age
# 2.Cabin
# 3.Embarked
# Cabin has more null values

# In[14]:


sns.heatmap(titanic.isnull())


# In[15]:


sns.boxplot(x="Pclass",y="Age",data=titanic)


# In[16]:


titanic.head()


# Dropping the Cabin column because it has large amount of missing values(Almost 75%)

# In[17]:


titanic.drop("Cabin",axis=1,inplace=True)


# In[18]:


#Removing cabin column because lot of lissing values
titanic.head()


# In[19]:


titanic.dropna(inplace=True)


# In[20]:


titanic.head()


# Let's get to know the count of all the missing values in the data

# In[21]:


titanic.isnull().sum()


# In[22]:


titanic


# While we are performing Logistic regression we don't deal with categorical values 
# But the data needed to be in more accurate way to model to creating dummy variables for categorical variables
# Sex has two values 0[Female] and 1[Male]
# so lets create dummy variable for male
# A 0 value at male column represents that it would be female

# In[23]:


sex=pd.get_dummies(titanic["Sex"],drop_first=True)
sex.head()


# In[24]:


embark=pd.get_dummies(titanic["Embarked"],drop_first=True)
embark.head()


# In[25]:


pcl=pd.get_dummies(titanic["Pclass"],drop_first=True)
pcl.head()


# In[26]:


titanic=pd.concat([titanic,sex,embark,pcl],axis=1)
titanic.head()


# Dropping all the string values and replacing them with the dummy values

# In[27]:


titanic.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[28]:


titanic.head()


# In[29]:


titanic.drop('Pclass',axis=1,inplace=True)


# In[30]:


titanic.head()


# Declaring the Survived as our y value
# and rest of data as out independant variable X

# Trainig and Testing data by splitting data
# 

# In[31]:


X=titanic.drop("Survived",axis=1)
y=titanic["Survived"]


# Imporing necessary modeles from library sklearn to perform logistic regression

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Dividing the data into test and train data
# Train data is basically used to develop the model and 
# model is tested on test data to predict y values
# 70% of data is used for training because we declared the test_size to 0.3

# In[38]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[40]:


logmodel=LogisticRegression()


# In[41]:


logmodel.fit(X_train,y_train)


# In[42]:


predictions=logmodel.predict(X_test)


# Classification report gives us the metrics like presicion , recall and also F-1 score of the model

# In[44]:


classification_report(y_test,predictions)


# In[46]:


confusion_matrix(y_test,predictions)


# In[49]:


accuracy_score(y_test,predictions)


# In[ ]:





# In[ ]:





# In[ ]:




