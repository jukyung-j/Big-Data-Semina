#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


# In[35]:


dataset = pd.read_excel('./AirQualityUCI.xlsx')


# In[36]:


dataset.head()


# CO(GT) : True hourly averaged concentration CO in mg/m^3 (reference analyzer)

# In[37]:


dataset.isnull().sum()


# In[38]:


X = dataset.iloc[:,:-1]
y = dataset.iloc[:,[-1]]
print(X)
print(y)


# In[39]:


sns.pairplot(dataset)
plt.show()


# In[40]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,test_size=0.2)


# In[41]:


correlation = dataset.corr()

plt.figure(figsize=(10,10))
sns.heatmap(correlation, annot=True,fmt = '.2f', cmap="Blues",square=True)


# In[50]:


from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
pr = LinearRegression()

mlr.fit(x_train,y_train)


# In[51]:


y_predict = mlr.predict(x_test)
print(mlr.coef_)
print(mlr.intercept_)


# In[63]:


print(x_train.shape)
print(y_train.shape)


# In[56]:


plt.title('AirQualityUCI')
plt.xlabel('acutal')
plt.ylabel('predict')
plt.grid(True)
plt.scatter(y_test,y_predict,alpha=0.4)
plt.show()


# accuracy

# In[19]:


from sklearn.metrics import mean_squared_error, r2_score

r2 = r2_score(y_test,y_predict)
print(r2)


# 성능 RMSE

# In[20]:



np.round(np.sqrt(mean_squared_error(y_test,y_predict)),2)


# In[ ]:





# In[ ]:




