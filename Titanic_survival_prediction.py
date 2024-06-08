#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#loading dataset
df = pd.read_csv(r"C:\Users\jithu\Downloads\Titanic-Dataset.csv")


# In[3]:


df.head()


# In[4]:


#data preprocessing
df.isnull().sum()


# In[5]:


import seaborn as sns

numeric_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix on numeric data
corr = numeric_df.corr()
# Create a heatmap
sns.heatmap(corr, annot=True)
plt.show()


# In[6]:


# replacing the age column with mean

df['Age'] = df['Age'].fillna(df['Age'].mean())
#droping cabin  column

df.drop(columns=['Cabin'], inplace=True)


# In[7]:


#  one hot encoding for sex
df = pd.get_dummies(df, columns=["Sex"], prefix=["Sex_"], dtype=int)


# In[8]:


# one hot encoding for EmbarkedS
df = pd.get_dummies(df, columns=["Embarked"], prefix=["Embarked_"], dtype=int)


# In[9]:


df.head()


# In[10]:


# defining feature and target variables
X=df[['Pclass','Age','SibSp','Parch','Fare','Sex__male','Embarked__C','Embarked__Q','Embarked__S']]
Y=df['Survived']


# In[11]:


# split df into training and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[12]:


#logistic regression model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)


# In[13]:


#prediction on test data
prediction=model.predict(X_test)


# In[14]:


prediction


# In[15]:


#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,prediction))


# In[16]:


#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))


# In[17]:


#creating data frame for a new person
person={'Pclass':1,'Age':30,'SibSp':1,'Parch':1,'Fare':80,'Sex__male':0,'Embarked__C':1,'Embarked__Q':0,'Embarked__S':0}
person_df=pd.DataFrame([person])
person_df



# In[18]:


#model  prediction for the new person
model.predict(person_df)

