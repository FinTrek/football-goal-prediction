#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#imputation
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
# train_test_split
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# feature selection
from sklearn.feature_selection import RFE
# classification models
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


missing_values = ["n/a", "na", "--"]
df = pd.read_csv("D:/DS/zs/zs_data.csv",index_col = 0, na_values = missing_values)


# In[4]:


#pandas_profiling.ProfileReport(df)


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df['shot_id_number'] = df.index + 1


# In[8]:


df['shot_id_number'].isnull().any()


# In[9]:


df.head(10)


# In[10]:


df['knockout_match'].value_counts()


# In[11]:


df['match_event_id'].value_counts().head()


# In[12]:


df['power_of_shot'].value_counts()


# In[13]:


correlation = df.corr()
sns.heatmap(correlation, cmap = 'viridis')


# In[14]:


correlation = df.corr()['is_goal']
# convert series to dataframe so it can be sorted
correlation_df = pd.DataFrame(correlation)
# correct column label from Points to correlation
correlation_df.columns = ["Correlation"]
# sort correlation
corr_sorted = correlation_df.sort_values(by=['Correlation'], ascending=False)
corr_sorted.head(20)


# In[15]:


df2 = df.loc[df['is_goal'].isnull() == True]


# In[16]:


df['is_goal'].isnull().sum()


# In[17]:


df1 = df.drop(df[df['is_goal'].isnull()].index)


# In[18]:


df1.shape


# In[19]:


df2.shape


# In[20]:


df.shape


# In[21]:


df1['is_goal'].isnull().any()


# In[22]:


df2['is_goal'].value_counts()


# Hence,
# * df is the main dataset
# * df1 is the df with non NULL is_goal values
# * df2 is the df with NULL is_goal values

# In[23]:


submit = df2[['shot_id_number', 'is_goal']].copy()


# In[24]:


submit.shape


# Checking for NULL values in other features of df1

# In[25]:


print(df1.isnull().any())


# In[26]:


df1.head()


# In[27]:


df1.count()


# In[28]:


df1.shape


# By comparing the above two cells, we find that apart from 'match_id' , 'team_id' , 'shot_id_number' & 'is_goal', all other features have NULL values.

# In[29]:


df1 = df1.drop(['match_event_id', 'location_x', 'location_y', 'game_season', 'team_name', 'date_of_game', 'lat/lng', 'match_id', 'team_id'], axis = 1)


# In[30]:


df1.head()


# Catgorical Variables: 
# * area_of_shot
# * shot_basics
# * home/away
# * type_of_shot
# * type_of_combined_shot

# Dropping knockout_match.1, remaining_min.1, power_of_shot.1 since they contain weird values

# In[31]:


df1 = df1.drop(['knockout_match.1', 'power_of_shot.1', 'remaining_min.1'], axis = 1)


# In[32]:


df1.head()


# In[33]:


df1['remaining_min'].value_counts()


# In[34]:


df1['power_of_shot'].value_counts()


# In[35]:


df1['knockout_match'].value_counts()


# In[36]:


df1['remaining_sec'].value_counts().head()


# In[37]:


df1['remaining_sec.1'].value_counts().head()


# In[38]:


df1['distance_of_shot'].value_counts().head()


# In[39]:


df1['distance_of_shot.1'].value_counts().head()


# In[40]:


df1['area_of_shot'].value_counts()


# In[41]:


df1['shot_basics'].value_counts()


# In[42]:


df1['range_of_shot'].value_counts()


# In[43]:


df1['home/away'].value_counts().head()


# In[44]:


df1['type_of_shot'].value_counts().head()


# In[45]:


df1['type_of_combined_shot'].value_counts()


# Numeric Variables:
# * remaining_min
# * power_of_shot
# * knockout_match
# * remaining_sec
# * remaining_sec.1
# * distance_of_shot
# * distance_of_shot.1
# 
# Categorical Variables:
# * area_of_shot
# * shot_basics
# * range_of_shot
# * home/away
# * type_of_shot
# * type_of_combined_shot

# In[46]:


df1.head()


# In[47]:


df1['home/away'] = df1['home/away'].fillna(method='ffill')


# In[48]:


df1['home/away'].isnull().any()


# In[49]:


ha = np.asarray(df1[['home/away']])


# In[50]:


ha.size


# In[131]:


df1['home/away'] = df1['home/away'].str.contains("@", regex = True)


# In[132]:


df1['home/away'].size


# In[134]:


df1['home/away'].head(20)


# ### IMPUTATION

# In[53]:


len(df1) - df1.count()


# In[54]:


df1.shape


# Imputation for remaining_min

# In[55]:


df1['remaining_min'] = df1['remaining_min'].fillna(method='ffill')


# In[56]:


df1['remaining_min'].isnull().any()


# In[57]:


df1['remaining_min'].count()


# Imputation for power_of_shot

# In[58]:


df1['power_of_shot'].value_counts()


# In[59]:


median = df1['power_of_shot'].median()
df1['power_of_shot'] = df1['power_of_shot'].fillna(median)


# In[60]:


df1['power_of_shot'].isnull().any()


# Imputation of knockout_match

# In[61]:


df1['knockout_match'] = df1['knockout_match'].fillna(method='ffill')


# In[62]:


df1['knockout_match'].isnull().any()


# Imputation of remaining_sec & remaining_sec.1

# In[63]:


m1 = df1['remaining_sec'].median()
df1['remaining_sec'] = df1['remaining_sec'].fillna(m1)
m2 = df1['remaining_sec.1'].median()
df1['remaining_sec.1'] = df1['remaining_sec.1'].fillna(m2)


# In[64]:


df1['remaining_sec'].isnull().any()


# In[65]:


df1['remaining_sec.1'].isnull().any()


# Imputation of distance_of_shot & distance_of_shot.1

# In[66]:


m1 = df1['distance_of_shot'].median()
df1['distance_of_shot'] = df1['distance_of_shot'].fillna(m1)
m2 = df1['distance_of_shot.1'].median()
df1['distance_of_shot.1'] = df1['distance_of_shot'].fillna(m2)


# In[67]:


df1['distance_of_shot'].isnull().any()


# In[68]:


df1['distance_of_shot.1'].isnull().any()


# Imputing Categorical Variables

# In[135]:


df1['area_of_shot'] = df1['area_of_shot'].fillna(df1['area_of_shot'].value_counts().index[0])
df1['shot_basics'] = df1['shot_basics'].fillna(df1['shot_basics'].value_counts().index[0])
df1['range_of_shot'] = df1['range_of_shot'].fillna(df1['range_of_shot'].value_counts().index[0])
df1['home/away'] = df1['home/away'].fillna(df1['home/away'].value_counts().index[0])
df1['type_of_shot'] = df1['type_of_shot'].fillna(df1['type_of_shot'].value_counts().index[0])
df1['type_of_combined_shot'] = df1['type_of_combined_shot'].fillna(df1['type_of_combined_shot'].value_counts().index[0])


# In[136]:


df1.isnull().any()


# Encoding Categorical Data

# Categorical Variables:
# * area_of_shot
# * shot_basics
# * range_of_shot
# * home/away
# * type_of_shot
# * type_of_combined_shot

# In[137]:


le = LabelEncoder()


# In[141]:


df1['area_of_shot'] = le.fit_transform(df1['area_of_shot'])
df1['shot_basics'] = le.fit_transform(df1['shot_basics'])
df1['range_of_shot'] = le.fit_transform(df1['range_of_shot'])
df1['home/away'] = le.fit_transform(df1['home/away'])
df1['type_of_shot'] = le.fit_transform(df1['type_of_shot'])
df1['type_of_combined_shot'] = le.fit_transform(df1['type_of_combined_shot'])


# In[145]:


X = df1.copy()


# In[146]:


X.head()


# In[147]:


y = X['is_goal'].copy()


# In[148]:


X = X.drop(['is_goal', 'shot_id_number'], axis = 1)


# In[149]:


X = X.drop(['remaining_sec.1', 'distance_of_shot.1'], axis = 1)


# In[150]:


X.head()


# In[151]:


y = pd.DataFrame(y)


# In[152]:


y.head()


# In[153]:


X.shape


# In[154]:


y.shape


# In[155]:


6268/24429


# In[214]:


X = np.asarray(X)


# In[215]:


y = np.asarray(y)


# In[266]:


skf = StratifiedKFold(n_splits=5, random_state = 42, shuffle = True)


# In[267]:


for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[ ]:





# In[268]:


1/0.25658029391


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[218]:


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25658029391, random_state = 42)


# In[269]:


from xgboost import XGBClassifier


# In[270]:


model = XGBClassifier()
model.fit(X_train, y_train)


# In[271]:


print(model.feature_importances_)


# In[272]:


X.head()


# In[273]:


rfm = RandomForestClassifier(n_jobs = 10, random_state = 42)
rfm.fit(X_train, y_train)
y_pred_rfm = rfm.predict(X_test)


# In[274]:


rfm.score(X_test, y_test)


# In[275]:


knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# In[276]:


knn.score(X_test, y_test)


# In[277]:


sgd = SGDClassifier(loss = 'modified_huber', shuffle = True, random_state = 101)
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)


# In[278]:


sgd.score(X_test, y_test)


# In[279]:


nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)


# In[280]:


nb.score(X_test, y_test)


# In[281]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)


# In[282]:


logreg.score(X_test, y_test)


# In[283]:


dtree = DecisionTreeClassifier(max_depth = 10, random_state = 10, max_features = None, min_samples_leaf = 15)
dtree.fit(X_train, y_train)
y_pred_dt = dtree.predict(X_test)


# In[284]:


dtree.score(X_test, y_test)


# In[285]:


print(classification_report(y_test, y_pred_dt))


# In[286]:


dtree_roc_auc = roc_auc_score(y_test, y_pred_dt)
fpr, tpr, thresholds = roc_curve(y_test, dtree.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree (area = %0.2f)' % dtree_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# #### Hyperparameter Optiization for Random Forest Classifier

# In[237]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 20, num = 10)]
max_depth = [int(x) for x in np.linspace(1, 50, num = 10)]
max_depth.append(None)
max_features = ['auto', 'sqrt']

param_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'max_features': max_features,
}

estimator = RandomForestClassifier(random_state = 69)
cv_test = KFold(n_splits=5)
gscv = GridSearchCV(estimator, param_grid, n_jobs = -1, 
                        scoring = 'roc_auc', cv = cv_test, 
                        verbose = 2)

gscv.fit(X_train, y_train)


# In[287]:


gscv.best_params_


# In[288]:


best_model = gscv.best_estimator_


# In[289]:


best_model.score(X_test,y_test)


# In[242]:


rf2_pred = best_model.predict(X_test)
rf2_prob = best_model.predict_proba(X_test)[:, 1]


# In[243]:


rf2_roc_auc = roc_auc_score(y_test, rf2_pred)
fpr, tpr, thresholds = roc_curve(y_test, rf2_prob)
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (Model Tuned) (area = %0.2f)' % rf2_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.04, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[290]:


y_pred_dt.size


# In[291]:


#prob = dtree.predict_proba(X_test)[:,1]


# In[292]:


#prob.size


# In[293]:


prob = pd.DataFrame(rf2_prob)


# In[294]:


prob.tail()


# In[295]:


prob['is_goal'] = prob[0] 


# In[296]:


prob.head()


# In[297]:


prob = prob.drop([0], axis = 1)


# In[298]:


prob['is_goal'].isnull().any()


# In[299]:


df2.shape


# In[300]:


prob.shape


# In[301]:


submit =  df2[['shot_id_number']].copy()


# In[302]:


submit.shape


# In[190]:


submit = submit.reset_index()


# In[191]:


submit.isnull().any()


# In[192]:


submit['is_goal'] = prob[['is_goal']].copy()


# In[193]:


submit.isnull().any()


# In[194]:


submit.to_csv(r'D:/DS/zs_submit.csv')


# In[195]:


get_ipython().system('jupyter nbconvert --to script zs.ipynb')

