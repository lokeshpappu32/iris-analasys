#!/usr/bin/env python
# coding: utf-8

# In[144]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[145]:


df=pd.read_csv('iris_flower.csv')
df


# In[146]:


df.describe()


# In[147]:


df.head()


# In[148]:


df.head(15)


# In[149]:


names=["SEPAL LENGTH","SEPAL WIDTH","PETAL LENGTH","PETAL WIDTH","CLASS LABELS"]


# In[150]:


df.tail()


# In[151]:


print(names)


# In[152]:


df.tail(5)


# In[153]:


df.info


# In[154]:


import pandas as pd
s=df["SEPAL LENGTH"]
print(s)


# In[155]:


import pandas as pd
s=df[0:2]
print(s)


# In[156]:


s=df.drop(1)
print(s)


# In[157]:


df.drop([ 'SEPAL LENGTH', 'SEPAL WIDTH'],axis=1)


# In[158]:


df.drop(['PETAL LENGTH','PETAL WIDTH' ],axis=1)


# In[159]:


df=pd.read_csv('iris_flower.csv')
df


# In[161]:


a=df.isnull()
print(a)


# In[162]:


s=df.fillna(0)
print(s)


# In[ ]:





# In[173]:


a=df.groupby("CLASS LABELS")[["SEPAL LENGTH","SEPAL WIDTH","PETAL LENGTH","PETAL WIDTH"]].count()


# In[118]:


a


# In[175]:


# Visualize the whole dataset
sns.pairplot(df, hue='CLASS LABELS')


# In[176]:


a.plot.bar(stacked=True)


# In[120]:


df.plot.bar(stacked=True)


# In[121]:


import matplotlib.pyplot as plt
x=['iris-setosa','iris-versicolor','iris-virginica']
y=[50,50,50]
plt.bar(x,y)


# In[122]:


import numpy as np
import pandas as pd
# Seperate features and target  
data = df.values
X = data[:,0:4]
Y = data[:,4]


# In[123]:


# Calculate avarage of each features for all classes
Y_Data =np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1]) for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4,3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(names)-1)
width = 0.25


# In[127]:


# Plot the avarage
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, names[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()


# In[128]:


# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[129]:


# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)


# In[130]:


# Predict from the test dataset
predictions = svn.predict(X_test)


# In[131]:


# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[132]:


# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[133]:


X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))


# In[134]:


# Save the model
import pickle
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)


# In[135]:


# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)


# In[136]:


model.predict(X_new)


# In[139]:


X_new = np.array([[5, 2, 2.2, 0.2], [  4.9, 3, 4.8, 1.1 ], [  5.3, 6, 4.6, 2.5]])
#Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))


# In[ ]:




