#!/usr/bin/env python
# coding: utf-8

# In[2]:


data_fake=pd.read_csv('Fake.csv')
data_true=pd.read_csv('True.csv')


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn. metrics import classification_report
import re
import string


# In[3]:


data_fake.head()


# In[4]:


data_true.head()


# In[5]:


data_fake['class']=0
data_true['class']=1


# In[6]:


data_fake.shape,data_true.shape


# In[7]:


data_fake_manual_testing=data_fake.tail(10)
for i in range(23480,23470,-1):   
    data_fake.drop([i],axis=0,inplace=True)

    data_true_manual_testing=data_true.tail(10)
for i  in range(21416,21406,-1):
    data_true.drop([i],axis=0,inplace=True)


# In[8]:


data_fake.shape,data_true.shape


# In[9]:


data_fake_manual_testing['class']=0
data_true_manual_testing['class']=1


# In[10]:


data_fake_manual_testing.head(20)


# In[11]:


data_true_manual_testing.head(20)


# In[12]:


data_merge=pd.concat([data_fake,data_true],axis=0)


# In[13]:


data_merge.head(10)


# In[14]:


data_merge.columns


# In[15]:


data = data_merge.drop(['title','subject', 'date'], axis = 1)


# In[16]:


data.isnull().sum()
data=data.dropna()
data.isnull().sum()


# In[17]:


data=data.sample(frac=1)


# In[18]:


data.reset_index(inplace=True)
data.drop(['index'],axis=1,inplace=True)


# In[19]:


data.columns


# In[20]:


data.head()


# In[21]:


def wordopt(text):
    
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation,),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text
data["text"]=data["text"].apply(wordopt)
x=data['text']
y=data['class']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[22]:


from numpy.lib.function_base import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer


# In[23]:


vectorization=TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)


# In[24]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(xv_train,y_train)
Pred_lr=LR.predict(xv_test)
LR.score(xv_test,y_test)
print(classification_report(y_test,Pred_lr))


# In[25]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Get the confusion matrix
cm = confusion_matrix(y_test, Pred_lr)

# Create a heatmap from the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Set the axis labels
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()


# In[29]:


LR.score(xv_test,y_test)


# In[26]:


from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(xv_train,y_train)
pred_dt=DT.predict(xv_test)
DT.score(xv_test,y_test)
print(classification_report(y_test,Pred_lr))


# In[27]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Get the confusion matrix
cm = confusion_matrix(y_test, pred_dt)

# Create a heatmap from the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Set the axis labels
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()


# In[28]:


DT.score(xv_test,y_test)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
GB=GradientBoostingClassifier(random_state=0)
GB.fit(xv_train,y_train)
pred_gb=GB.predict(xv_test)
GB.score(xv_test,y_test)
print(classification_report(y_test,pred_gb))


# In[38]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Get the confusion matrix
cm = confusion_matrix(y_test, pred_gb)

# Create a heatmap from the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Set the axis labels
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()


# In[39]:


GB.score(xv_test,y_test)


# In[40]:


from sklearn.ensemble import RandomForestClassifier 
RF=RandomForestClassifier(random_state=0)
RF.fit(xv_train,y_train)
pred_rf=RF.predict(xv_test)
RF.score(xv_test,y_test)
print(classification_report(y_test,pred_rf))


# In[ ]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Get the confusion matrix
cm = confusion_matrix(y_test, pred_rf)

# Create a heatmap from the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Set the axis labels
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

plt.show()


# In[ ]:


RF.score(xv_test,y_test)


# In[ ]:


def output_lable(n):
    if n==0:
        return "Fake NEW"
    elif n==1:
        return "Not A Fake New"

def manual_testing(news):
    

    testing_news = {"text": [news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(wordopt)
    new_x_test=new_def_test['text']
    new_xv_test=vectorization.transform(new_x_test)
    pred_LR=LR.predict(new_xv_test)
    pred_DT=DT.predict(new_xv_test)
    pred_GB=GB.predict(new_xv_test)
    pred_RF=RF.predict(new_xv_test)
    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction:{}".format(output_lable(pred_LR[0]),
                                                                                                             output_lable(pred_DT[0]),
                                                                                                             output_lable(pred_GB[0]),
                                                                                                             output_lable(pred_RF[0])))


# In[ ]:


news=str(input())
manual_testing(news)


# In[ ]:




