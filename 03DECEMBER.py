#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


train=pd.read_excel('C:/Users/chandu prakash/Downloads/Data_train_new2.xlsx')
test=pd.read_excel('C:/Users/chandu prakash/Downloads/Data_test_new2.xlsx')


# In[3]:


train.head()


# In[4]:


train.isnull().sum()


# In[3]:


train["Average_Cost"]= train["Average_Cost"].replace('for','s200')


# In[4]:


import re
import string
def clean_text_round2(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('%s' % re.escape(string.punctuation),'',text)
    text=re.sub(r'\(.*\)', '',text)
    
    text=re.sub(':%s' % re.escape(string.punctuation),'',text)
    text=re.sub("\D", "", text)
    
    return text
round2=lambda x: clean_text_round2(x)
train['Average_Cost']=train.Average_Cost.apply(round2)
train['Minimum_Order']=train.Minimum_Order.apply(round2)
test['Average_Cost']=test.Average_Cost.apply(round2)
test['Minimum_Order']=test.Minimum_Order.apply(round2)


# In[5]:


train['Cuisines'] = train['Cuisines'].str.count(',') + 1
test['Cuisines'] = test['Cuisines'].str.count(',') + 1


# In[6]:


train["Minimum_Order"]=train.Minimum_Order.astype(int)
train["Average_Cost"]=train.Average_Cost.astype(int)
test["Minimum_Order"]=test.Minimum_Order.astype(int)
test["Average_Cost"]=test.Average_Cost.astype(int)


# In[7]:


train["Rating"]=train.Rating.astype(float)
train["Votes"]=train.Votes.astype(int)
train["Reviews"]=train.Reviews.astype(int)
test["Rating"]=test.Rating.astype(float)
test["Votes"]=test.Votes.astype(int)
test["Reviews"]=test.Reviews.astype(int)


# In[8]:


location=pd.get_dummies(train.Location,drop_first=True)


# In[9]:


location1=pd.get_dummies(test.Location,drop_first=True)


# In[10]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Delivery_Time= le.fit_transform(train.Delivery_Time)


# In[11]:


final=pd.DataFrame(train,columns=['Cuisines','Minimum_Order','Average_Cost','Votes','Reviews','Rating'])


# In[12]:


final1=pd.DataFrame(test,columns=['Cuisines','Minimum_Order','Average_Cost','Votes','Reviews','Rating'])


# In[38]:


Notscale=pd.concat([final],axis=1,join='outer')
X=Notscale
Y=pd.DataFrame(train,columns=['Delivery_Time'])


# In[39]:


test1=pd.concat([final1],axis=1,join='outer')


# In[17]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[18]:


import xgboost


# In[19]:


xgb1 = xgboost.XGBClassifier()


# In[20]:


## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[40]:


random_search=RandomizedSearchCV(xgb1,param_distributions=params,n_iter=10
                                 ,n_jobs=-1,cv=10,verbose=3)


# In[41]:


random_search.fit(X,Y)


# In[42]:


random_search.best_estimator_


# In[149]:


xgb1 = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.8,
              colsample_bytree=0.8, gamma=0.0, learning_rate=0.1,
              max_delta_step=0, max_depth=15, min_child_weight=1, missing=None,
              n_estimators=100, n_jobs=4, nthread=4,
              objective='multi:softprob', random_state=100, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=5, seed=37, silent=True,
              subsample=0.8)


# In[150]:


from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
seed = 8
skf=StratifiedKFold(n_splits=10, random_state=None)
from sklearn.model_selection import cross_val_score
seed = 8
kfold = model_selection.KFold(n_splits=8, random_state=seed)
score=cross_val_score(xgb1,X,Y,cv=skf)


# In[151]:


score


# In[152]:


score.mean()


# In[153]:


xgb1.fit(X,Y)


# In[154]:


result1=xgb1.predict(test1)
np.unique(result1,return_counts=True)


# In[125]:


an1=pd.DataFrame(result1,columns=['Delivery_Time'])


# In[126]:


an1.to_excel("26novxg42.xlsx")


# In[ ]:




