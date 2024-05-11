import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your data
# data = pd.read_csv('your_data.csv')

# Assume 'data' is your DataFrame and is already prepared

#split data
X = data.drop('target_column', axis=1)  # Features
y = data['target_column']               # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#light gbm dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

# Database configuration for storing study results
mysqldb = 'mysql+pymysql://root:Hawaii808!@localhost:3306/bank_churn_lightgbm_history'

def objective(trial):
    # Hyperparameters to be tuned by Optuna
    light_param = {
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 16, 22),
        'max_depth': trial.suggest_int('max_depth', 38, 49),
        'learning_rate': trial.suggest_float('learning_rate', 0.020, 0.026),
        'n_estimators': trial.suggest_int('n_estimators', 2800, 3000),
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 66000, 68000),
        'min_child_weight': trial.suggest_int('min_child_weight', 4, 6),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 40),
        'subsample': trial.suggest_float('subsample', 0.94, 0.99),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.35, 0.42),
        'reg_alpha': trial.suggest_float('reg_alpha', 3.5, 4.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.9, 1.0),
        'random_state': trial.suggest_categorical('random_state', [None, 42]),
        'n_jobs': -1
    }
    
    # Initialize the model with the parameters for regression
    model = LGBMRegressor(**light_param)
    
    # Perform cross-validation using RMSE as the scoring metric
    kfold = KFold(n_splits=7, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
    
    # The objective value is the average of the negative RMSE scores
    return np.mean(scores)

# Create a study object and specify the direction as 'maximize' because neg_rmse is negative.
study = optuna.create_study(direction='maximize', study_name='Bank_Churn_Testing', storage=mysqldb, load_if_exists=True)

# Optimize the study
study.optimize(objective, n_trials=1)

# Display the best parameters
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Retrain the model with the best parameters found
best_params = study.best_params
best_model = LGBMRegressor(**best_params)
best_model.fit(X, y)

# Assuming you have a test DataFrame `df_test`
df_test_features = df_test[target_features]  # Assuming 'target_features' is defined

# Make predictions
predictions = best_model.predict(df_test_features)

# Optional: Further steps to handle predictions, e.g., save to CSV or adjust submission format, etc.
print("Predictions are ready.")
