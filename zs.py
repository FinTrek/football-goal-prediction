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
import warnings
warnings.filterwarnings("ignore")

missing_values = ["n/a", "na", "--"]
df = pd.read_csv("D:/DS/zs/zs_data.csv",index_col = 0, na_values = missing_values)


pandas_profiling.ProfileReport(df)

print(df.describe())
print(df.info())

df['shot_id_number'] = df.index + 1

print(df['shot_id_number'].isnull().any())

print(df['knockout_match'].value_counts())

print(df['match_event_id'].value_counts().head())

print(df['power_of_shot'].value_counts())

correlation = df.corr()
sns.heatmap(correlation, cmap = 'viridis')

correlation = df.corr()['is_goal']
# convert series to dataframe so it can be sorted
correlation_df = pd.DataFrame(correlation)
# correct column label from Points to correlation
correlation_df.columns = ["Correlation"]
# sort correlation
corr_sorted = correlation_df.sort_values(by=['Correlation'], ascending=False)
corr_sorted.head(20)

df2 = df.loc[df['is_goal'].isnull() == True]

df['is_goal'].isnull().sum()

df1 = df.drop(df[df['is_goal'].isnull()].index)

print(df1.shape)
print(df2.shape)
print(df.shape)


# Hence,
# * df is the main dataset
# * df1 is the df with non NULL is_goal values
# * df2 is the df with NULL is_goal values

submit = df2[['shot_id_number', 'is_goal']].copy()

print(df1.isnull().any())

df1 = df1.drop(['match_event_id', 'location_x', 'location_y', 'game_season', 'team_name', 'date_of_game', 'lat/lng', 'match_id', 'team_id'], axis = 1)


# Catgorical Variables: 
# * area_of_shot
# * shot_basics
# * home/away
# * type_of_shot
# * type_of_combined_shot

# Dropping knockout_match.1, remaining_min.1, power_of_shot.1 since they contain weird values

df1 = df1.drop(['knockout_match.1', 'power_of_shot.1', 'remaining_min.1'], axis = 1)


print(df1['remaining_min'].value_counts())
print(df1['power_of_shot'].value_counts())
print(df1['knockout_match'].value_counts())
print(df1['remaining_sec'].value_counts().head())
print(df1['remaining_sec.1'].value_counts().head())
print(df1['distance_of_shot'].value_counts().head())
print(df1['distance_of_shot.1'].value_counts().head())
print(df1['area_of_shot'].value_counts())
print(df1['shot_basics'].value_counts())
print(df1['range_of_shot'].value_counts())
print(df1['home/away'].value_counts().head())
print(df1['type_of_shot'].value_counts().head())
print(df1['type_of_combined_shot'].value_counts())


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


#mask1 = ('@' in df1['home/away'] == True)



#df1['home/away'] = np.where(mask, df['GAME'], df['EVENT'])


"""
for row in df1:
    if('@' in row):
        row = 'AWAY'
    elif('vs.' in row):
        row = 'HOME'
"""




#df1['home/away'].head()


# ### IMPUTATION

# In[50]:


len(df1) - df1.count()


# In[51]:


df1.shape


# Imputation for remaining_min

# In[52]:


df1['remaining_min'] = df1['remaining_min'].fillna(method='ffill')


# In[53]:


df1['remaining_min'].isnull().any()


# In[54]:


df1['remaining_min'].count()


# Imputation for power_of_shot

# In[55]:


df1['power_of_shot'].value_counts()


# In[56]:


median = df1['power_of_shot'].median()
df1['power_of_shot'] = df1['power_of_shot'].fillna(median)


# In[57]:


df1['power_of_shot'].isnull().any()


# Imputation of knockout_match

# In[58]:


df1['knockout_match'] = df1['knockout_match'].fillna(method='ffill')


# In[59]:


df1['knockout_match'].isnull().any()


# Imputation of remaining_sec & remaining_sec.1

# In[60]:


m1 = df1['remaining_sec'].median()
df1['remaining_sec'] = df1['remaining_sec'].fillna(m1)
m2 = df1['remaining_sec.1'].median()
df1['remaining_sec.1'] = df1['remaining_sec.1'].fillna(m2)


# In[61]:


df1['remaining_sec'].isnull().any()


# In[62]:


df1['remaining_sec.1'].isnull().any()


# Imputation of distance_of_shot & distance_of_shot.1

# In[63]:


m1 = df1['distance_of_shot'].median()
df1['distance_of_shot'] = df1['distance_of_shot'].fillna(m1)
m2 = df1['distance_of_shot.1'].median()
df1['distance_of_shot.1'] = df1['distance_of_shot'].fillna(m2)


# In[64]:


df1['distance_of_shot'].isnull().any()


# In[65]:


df1['distance_of_shot.1'].isnull().any()


# Imputing Categorical Variables

# In[66]:


df1['area_of_shot'] = df1['area_of_shot'].fillna(df1['area_of_shot'].value_counts().index[0])
df1['shot_basics'] = df1['shot_basics'].fillna(df1['shot_basics'].value_counts().index[0])
df1['range_of_shot'] = df1['range_of_shot'].fillna(df1['range_of_shot'].value_counts().index[0])
df1['home/away'] = df1['home/away'].fillna(df1['home/away'].value_counts().index[0])
df1['type_of_shot'] = df1['type_of_shot'].fillna(df1['type_of_shot'].value_counts().index[0])
df1['type_of_combined_shot'] = df1['type_of_combined_shot'].fillna(df1['type_of_combined_shot'].value_counts().index[0])


# In[67]:


df1.isnull().any()


# Encoding Categorical Data

# Categorical Variables:
# * area_of_shot
# * shot_basics
# * range_of_shot
# * home/away
# * type_of_shot
# * type_of_combined_shot

# In[68]:


le = LabelEncoder()


# In[69]:


df1['area_of_shot'] = le.fit_transform(df1['area_of_shot'])
df1['shot_basics'] = le.fit_transform(df1['shot_basics'])
df1['range_of_shot'] = le.fit_transform(df1['range_of_shot'])
#df1['home/away'] = le.fit_transform(df1['home/']) // HOME/AWAY CONSIDER LATER
df1['type_of_shot'] = le.fit_transform(df1['type_of_shot'])
df1['type_of_combined_shot'] = le.fit_transform(df1['type_of_combined_shot'])


# In[70]:


X = df1.copy()


# In[71]:


X.head()


# In[72]:


X = X.drop(['home/away'], axis = 1)
#later


# In[73]:


y = X['is_goal'].copy()


# In[74]:


X = X.drop(['is_goal', 'shot_id_number'], axis = 1)


# In[75]:


X = X.drop(['remaining_sec.1', 'distance_of_shot.1'], axis = 1)


# In[76]:


X.head()


# In[77]:


y = pd.DataFrame(y)


# In[78]:


y.head()


# In[79]:


X.shape


# In[80]:


y.shape


# In[81]:


6268/24429


# In[82]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25658029391, random_state = 42)


# In[83]:


rfm = RandomForestClassifier(n_jobs = 10, random_state = 42)
rfm.fit(X_train, y_train)
y_pred_rfm = rfm.predict(X_test)


# In[84]:


rfm.score(X_test, y_test)


# In[85]:


knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)


# In[86]:


knn.score(X_test, y_test)


# In[87]:


sgd = SGDClassifier(loss = 'modified_huber', shuffle = True, random_state = 101)
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)


# In[88]:


sgd.score(X_test, y_test)


# In[89]:


nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)


# In[90]:


nb.score(X_test, y_test)


# In[91]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)


# In[92]:


logreg.score(X_test, y_test)


# In[93]:


dtree = DecisionTreeClassifier(max_depth = 10, random_state = 10, max_features = None, min_samples_leaf = 15)
dtree.fit(X_train, y_train)
y_pred_dt = dtree.predict(X_test)


# In[94]:


dtree.score(X_test, y_test)


# In[95]:


print(classification_report(y_test, y_pred_dt))


# In[96]:


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

# In[ ]:


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


# In[ ]:


gscv.best_params_


# In[ ]:


best_model = gscv.best_estimator_


# In[ ]:


best_model.score(X_test,y_test)


# In[ ]:


rf2_pred = best_model.predict(X_test)
rf2_prob = best_model.predict_proba(X_test)[:, 1]


# In[ ]:


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


# In[ ]:


y_pred_dt.size


# In[ ]:


prob = dtree.predict_proba(X_test)[:,1]


# In[ ]:


prob.size


# In[ ]:


prob = pd.DataFrame(prob)


# In[ ]:


prob.tail()


# In[ ]:


prob['is_goal'] = prob[0] 


# In[ ]:


prob.head()


# In[ ]:


prob = prob.drop([0], axis = 1)


# In[ ]:


prob['is_goal'].isnull().any()


# In[ ]:


df2.shape


# In[ ]:


prob.shape


# In[ ]:


submit =  df2[['shot_id_number']].copy()


# In[ ]:


submit.shape


# In[ ]:


submit = submit.reset_index()


# In[ ]:


submit.isnull().any()


# In[ ]:


submit['is_goal'] = prob[['is_goal']].copy()


# In[ ]:


submit.isnull().any()


# In[ ]:


submit.to_csv(r'D:/DS/zs_submit.csv')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script zs.ipynb')

