#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC


from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")


# # ExploreData

# In[2]:


train=pd.read_csv(r"E:\titanic\train.csv")
test=pd.read_csv(r"E:\titanic\test.csv")


# In[3]:


#train.head()
#train.tail()
train.sample(10)


# In[4]:


train.shape


# In[5]:


train.columns.tolist()


# In[6]:


train.nunique()


# In[7]:


train.info()


# In[8]:


sns.heatmap(train.isnull())


# In[9]:


def clean(D):
    D.drop(['Cabin','Name','Ticket','Embarked','Fare'],axis=1,inplace=True)
    D.Age=D.Age.fillna(D.Age.median())
    D.dropna()
    return D


# In[10]:


clean(train)


# In[11]:


clean(test)


# In[12]:


sns.heatmap(train.isnull())


# # DataAnalysis

# In[13]:


train=train.drop('Sex',axis=1)


# In[14]:


train


# In[15]:


train.corr()


# In[16]:


co=train.corr()


# In[17]:


sns.heatmap(co,annot=True,fmt='.2f',linewidth=0.5,cmap='Pastel2', linewidths=2)


# In[18]:


train=pd.read_csv(r"E:\titanic\train.csv")


# In[19]:


train.Survived.value_counts()


# In[20]:


train.Sex.value_counts()


# In[21]:


train.Sex.value_counts().plot.pie(autopct='%0.2f%%')


# In[22]:


# Assuming 'df' is your DataFrame
quality_counts = train['Sex'].value_counts()
# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts,color=(0.545, 0.000, 0.545))
plt.title('Count Plot of Quality')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()


# In[23]:


# Assuming 'df' is your DataFrame
quality_counts = train['Pclass'].value_counts()
# Using Matplotlib to create a count plot
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts,color=(0.545, 0.000,0.545))
plt.title('Count Plot of Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()


# In[24]:


sns.histplot(train.Age)


# In[25]:


sns.catplot(x ="Sex", hue ="Survived", 
kind ="count", data = train)


# In[26]:


# It can be concluded that if a passenger paid a higher fare, the survival rate is more.
sns.catplot(x ='Embarked', hue ='Survived', 
kind ='count', col ='Pclass', data = train)


# In[27]:


#It helps in determining if higher-class passengers had more survival rate than the lower class ones or vice versa.
#Class 1 passengers have a higher survival chance compared to classes 2 and 3. 
#It implies that Pclass contributes a lot to a passenger’s survival rate.


# In[28]:


# Violinplot Displays distribution of data 
# across all levels of a category.
sns.violinplot(x ="Sex", y ="Age", hue ="Survived", 
data = train, split = True)


# # Transform Data

# In[29]:


#train.Sex=pd.get_dummies(train.Sex)


# In[30]:


train


# In[31]:


def clean(D):
    D.drop(['Cabin','Name','Ticket','Embarked','Fare'],axis=1,inplace=True)
    D.Age=D.Age.fillna(D.Age.median())
    D.dropna()
    return D


# In[32]:


clean(train)


# In[33]:


Sex_mapping = {'male': 0, 'female': 1}
train['Sex'] = train['Sex'].map(Sex_mapping)


# In[34]:


Sex_mapping = {'male': 0, 'female': 1}
test['Sex'] = test['Sex'].map(Sex_mapping)


# In[35]:


train


# In[36]:


test


# # Creat Model

# In[37]:


x=train.drop(['Survived'],axis=1)
y=train.Survived


# In[38]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)


# In[39]:


accuracies=[]


# In[40]:


def all(model):
    model.fit(x_train,y_train)
    mod=model1.predict(x_test)
    accuracy=accuracy_score(mod,y_test)
    print('Accuracy = ',accuracy)
    accuracies.append(accuracy)


# In[41]:


model1=LogisticRegression()
all(model1)


# In[42]:


model2=RandomForestClassifier()
all(model2)


# In[43]:


model3=GradientBoostingClassifier()
all(model3)


# In[44]:


model4=DecisionTreeClassifier()
all(model4)


# In[45]:


model5=KNeighborsClassifier()
all(model5)


# In[46]:


model6=GaussianNB()
all(model6)


# In[47]:


model7=SVC()
all(model7)


# In[48]:


Algorithmes=['RandomForestClassifier','RandomForestClassifier','GradientBoostingClassifier','DecisionTreeClassifier','KNeighborsClassifier','GaussianNB','SVC']


# In[49]:


New=pd.DataFrame({'Algorithmes':Algorithmes,'accuracies':accuracies})


# In[50]:


New


# In[51]:


modelx=GaussianNB()
modelx.fit(x_train,y_train)


# In[52]:


lastpredict=modelx.predict(test)


# In[53]:


final=test.PassengerId


# In[54]:


final_dataframe=pd.DataFrame({'PassengerId':final,'Survived':lastpredict})


# In[55]:


final_dataframe


# In[ ]:




