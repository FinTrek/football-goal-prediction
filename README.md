# Football-Goal-Prediction
Predicting whether Cristiano Ronaldo will score a goal based on a number of features

## 1.Data Preparation:

* Quality Checks: 
Some feature vectors were eliminated since they had nonsensical values especially the ones suffixed with ‘.1’ (knockout_match.1 and remaining_min.1). Other features were eliminated based on common sense relevance to the current problem. Some features like team_name and game_season have inaccurate values. For e.g., Ronaldo wasn’t playing professionally during the 1999 – 2000 season, yet the data said so. Also, during the 2011 – 12 season, he was playing for Real Madrid but the data says Manchester United. These kinds of features were eliminated from the get go.

* Data Preprocessing: 
Some of the values for shot_id_number were missing; they were imputed with the increment of the index variable. The original dataframe was split into two separate dataframes one with non-NULL values (df1) of is_goal and one with NULL values of is_goal(df2). Df1 has around ~24k values. The test size for train_test_split was set as the size of df2. Df2 was to be the main dataframe for all analysis since all values of (df2.is_goal) are non-NULL values. The relevant categorical features were label encoded in order to pass it through the various classification algorithms.

## 2.EDA:

* Summary Statistics and Profile Reporting using pandas_profiling
* Correlation Heat maps
* Other majority values for each feature were obtained by value_counts to gauge the distribution of variables among each feature in the dataset.

## 3.Model Building:

* Almost all classification models were tested with average scores.
* Hyperparameter tuning was done with GridSearchCV for RandomForestClassifier but yielded mediocre results again.
* Second Iteration:
* Stratified K Fold Cross Validation was used which resulted in minor performance improvement. But the requisite number of target variable probabilities (for is_goal) were not obtained since StratifiedKFold doesn’t allow for floating point parameter for test_size segregation.

## 4.Conclusion:
 
Considerations for possible score improvement:
•	Custom Feature Engineering
•	One Hot Encoding
•	Normalizing
•	Imputation using ML Models
•	Obtain multitude of features then XGBoost, RFE, Stability Selection for Feature Importance
•	Feature Hashing Schemes
•	Scaling
•	Deep(?)
