#!/usr/bin/env python
# coding: utf-8

# # In this notebook we try to practice all the classification algorithms that we learned in this course.
# # We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# # Lets first load required libraries:

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# # Lets download the data sets

# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# # Load data from csv file

# In[4]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[5]:


df.shape


# # Convert date time to object

# In[8]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data Visualization and Pre-Processing

# Let’s see how many of each class is in our data set

# In[10]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection

# Lets plot some columns to underestand data better:

# In[11]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# # All requested packages already installed.

# In[12]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[13]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing: Feature selection/extraction

# ## Lets look at the day of the week people get the loan

# In[14]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# In[15]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# # Convert Categorical features to numerical values

# Lets look at gender:

# In[16]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan

# Lets convert male to 0 and female to 1:

# In[17]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# # One Hot Encoding

# How about education?

# In[18]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# Feature befor One Hot Encoding

# In[19]:


df[['Principal','terms','age','Gender','education']].head()


# Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame

# In[20]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


#  Feature selection

# Lets defind feature sets, X:

# In[22]:


X = Feature
X[0:5] 


# What are our lables?

# In[23]:


y = df['loan_status'].values
y[0:5]


# # Normalize Data

# Data Standardization give data zero mean and unit variance (technically should be done after train test split

# In[24]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model You should use the following algorithm:

# K Nearest Neighbor(KNN)
# 
# Decision Tree
# 
# Support Vector Machine
# 
# Logistic Regression

# ## Notice:

# You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# 
# You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# 
# You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)

# Notice: You should find the best k to build the model with the best accuracy.
# 
# warning: You should not use the loan_test.csv for finding the best k, however, you can split your train_loan.csv into train and test to find the best k.

# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)


# In[27]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
Ks = 10
mean_acc = np.zeros((Ks-1)) 
std_acc = np.zeros((Ks-1)) 
ConfustionMx = [];
for n in range(1,Ks):
    #Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha= 0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
k0 = 1 + pd.Series(mean_acc).idxmax()
k0
mean_acc


# In[28]:


from sklearn.neighbors import KNeighborsClassifier
k=k0
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X,y)


# # Decision Tree

# In[29]:


from sklearn.tree import DecisionTreeClassifier
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X,y)


# ## Support Vector Machine

# In[30]:


from sklearn import svm


# In[31]:


clf = svm.SVC(kernel='rbf') 
clf.fit(X, y)


# # Logistic Regression

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.1, solver='liblinear').fit(X,y)


# # Model Evaluation using Test set

# In[34]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[35]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# # Load Test set for evaluation

# In[37]:


test_df = pd.read_csv('loan_test.csv')
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = df['effective_date'].dt.dayofweek
test_df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True) 
Feat = test_df[['Principal','terms','age','Gender','weekend']]
yreal = test_df['loan_status'].values
Feat = pd.concat([Feat,pd.get_dummies(test_df['education'])], axis=1)
Feat.drop(['Master or Above'], axis = 1,inplace=True)
test_X = preprocessing.StandardScaler().fit(Feat).transform(Feat)
#yhat for KNN
test_yhat = neigh.predict(test_X)
#yhat for decision tree
predTree = drugTree.predict(test_X)
#yhat for SVM
SVM_yhat = clf.predict(test_X)
SVM_yhat
#yhat for Linear
LR_yhat = LR.predict(test_X)
LR_yhat_prob = LR.predict_proba(test_X)


# In[38]:


KNN_jaccard = jaccard_similarity_score(yreal, test_yhat)
KNN_F1 = f1_score(yreal, test_yhat, average='weighted')
print(KNN_jaccard, KNN_F1)


# In[39]:


DT_jaccard = jaccard_similarity_score(yreal, predTree)
DT_F1 = f1_score(yreal, predTree, average='weighted')
print(DT_jaccard, DT_F1)


# In[40]:


SVM_jaccard = jaccard_similarity_score(yreal, SVM_yhat)
SVM_F1 = f1_score(yreal, SVM_yhat, average='weighted')
print(SVM_jaccard, SVM_F1)

LR_jaccard = jaccard_similarity_score(yreal, LR_yhat)
LR_F1 = f1_score(yreal, LR_yhat, average='weighted')
from sklearn.metrics import log_loss 
LL = log_loss(yreal, LR_yhat_prob)
print(LR_jaccard, LR_F1, LL)


# # Report

# You should be able to report the accuracy of the built model using different evaluation metrics:

# Algorithm	       Jaccard	F1-score	LogLoss
# KNN	                0.666	    0.632	    NA

# Decision Tree       0.759	    0.714	    NA

# SVM                0.722	    0.621	    NA

# LogisticRegression	0.740	    0.630	  0.549

# # Want to learn more?

# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to
# decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: SPSS
# Modeler

# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by
# data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to
# collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at Watson Studio

# In[ ]:




