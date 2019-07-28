# Progression Log

#### Current Score : 0.852 (Competition Evaluation)

### Mark 1

* Some feature vectors were eliminated since they had nonsensical values especially the ones suffixed with ‘.1’ (knockout_match.1 and remaining_min.1). Other features were eliminated based on common sense relevance to the current problem. Some features like team_name and game_season have inaccurate values. For e.g., Ronaldo wasn’t playing professionally during the 1999 – 2000 season, yet the data said so. Also, during the 2011 – 12 season, he was playing for Real Madrid but the data says Manchester United. These kinds of features were eliminated from the get go.
* Some of the values for shot_id_number were missing; they were imputed with the increment of the index variable. The original dataframe was split into two separate dataframes one with non-NULL values (df1) of is_goal and one with NULL values of is_goal(df2). Df1 has around ~24k values. The test size for train_test_split was set as the size of df2. Df2 was to be the main dataframe for all analysis since all values of df2.is_goal are non-NULL in nature. The relevant categorical features were label encoded in order to pass it through the various classification algorithms. 
* Summary Statistics and Profile Reporting using pandas_profiling
* Correlation Heat maps
* Other majority values for each feature were obtained by value_counts to gauge the distribution of variables among each feature in the dataset.

### Mark 2

* Hyperparameter tuning was done with GridSearchCV for RandomForestClassifier but yielded mediocre results again.

### Mark 3

* Stratified K Fold Cross Validation was used which resulted in minor performance improvement. But the requisite number of target variable probabilities (for is_goal) were not obtained since StratifiedKFold doesn’t allow for floating point parameter for test_size segregation.
* One Hot Encoding v Label Encoding on categorical feature vectors (Minor negligible Improvement).

### Mark 4

* Polynomial Expansion of the resultant dataframe of predictor features (No Improvement).   
* Standard Scaling of all feature vectors (No Improvement). 
* Principal Component Analysis [to reduce Dimensionality after Polynomial Expansion] (No Improvement).
      
### Further Work:

*	Custom Feature Engineering -> (Deep Feature Synthesis)
*	XGBoost, Recursive Feature Elimination, Stability Selection for Feature Importance
*	Anomaly Detection and Removal
*	Imputation using ML Models
*	Feature Hashing Schemes
*	Neural Network Implementation
