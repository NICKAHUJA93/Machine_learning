#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
pd.set_option('display.max_rows', 500)


# ### Reading Data

# In[2]:


csv_path ="E:\data.csv"
#Read the csv file and convert into pandas 
data = pd.read_csv(csv_path)


# In[3]:


#display first 10 row
data.head(10)


# ### Null handling 

# In[4]:


data.isnull().sum()
#data have no nulls :)


# ### Creating and Visualization correlation matrix 

# In[5]:


data.drop("id", axis =1, inplace = True)
corr = data.corr()
corr.style.background_gradient(cmap='coolwarm')


# ### Top 20 impacting features based on correlation 

# In[6]:


c = data.corr().abs()

s = c.unstack()
so = s.sort_values(kind="quicksort", ascending = False)

so = pd.DataFrame(so).reset_index(drop=False)
so.columns = ["label","Feature","Correlation"]

so = so[so["label"]=="label"].reset_index(drop = True)

so = so.head(20)


# In[7]:


so


# ### Train Test split

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(data.drop("label",axis=1), data["label"], test_size=0.33, random_state=42)


# ### Creating grid for random forest model

# In[9]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}


# ### Running the Grid

# In[10]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[11]:


rf_random.best_params_


# ### evaluate the model

# In[ ]:


evaluate(rf_random,X_test,y_test)


# In[ ]:





# In[ ]:




