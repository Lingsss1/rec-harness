
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import numpy as np
import os

# Define the data directory
DATA_DIR = './input'

# --- Load and Preprocess Data ---
def load_and_preprocess_data(file_path, is_train=True):
    df = pd.read_csv(file_path)

    # Data cleaning
    if is_train:
        df = df.dropna()
        df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 100)]
        df = df[
            (df['pickup_latitude'] > -90) & (df['pickup_latitude'] < 90) &
            (df['dropoff_latitude'] > -90) & (df['dropoff_latitude'] < 90) &
            (df['pickup_longitude'] > -180) & (df['pickup_longitude'] < 180) &
            (df['dropoff_longitude'] > -180) & (df['dropoff_longitude'] < 180)
        ]
        df = df[df['passenger_count'] > 0]
        df = df[df['passenger_count'] <= 6]
    else:
        #No data cleaning on test data as requested in prompt
        pass
        #df = df[df['passenger_count'] > 0]
        #df = df[df['passenger_count'] <= 6]

    # Feature Engineering
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Radius of Earth in kilometers
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    df['distance'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                                    df['dropoff_latitude'], df['dropoff_longitude'])

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day

    return df

# Load the training data
train_df = load_and_preprocess_data(os.path.join(DATA_DIR, 'train.csv'))

# Define features and target
features_lgbm = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance', 'hour', 'dayofweek']
features_catboost = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
            'passenger_count', 'distance', 'year', 'month', 'day', 'hour', 'dayofweek']
target = 'fare_amount'

# Split data into training and validation sets
X_lgbm = train_df[features_lgbm]
X_catboost = train_df[features_catboost]
y = train_df[target]

X_train_lgbm, X_val_lgbm, y_train_lgbm, y_val_lgbm = train_test_split(X_lgbm, y, test_size=0.2, random_state=42)
X_train_catboost, X_val_catboost, y_train_catboost, y_val_catboost = train_test_split(X_catboost, y, test_size=0.2, random_state=42)


# --- Model Training ---
# LightGBM Model
model_lgbm = LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05)
model_lgbm.fit(X_train_lgbm, y_train_lgbm)
y_pred_lgbm = model_lgbm.predict(X_val_lgbm)
rmse_lgbm = np.sqrt(mean_squared_error(y_val_lgbm, y_pred_lgbm))

# CatBoost Model
model_catboost = CatBoostRegressor(iterations=1000,  # Increased iterations
                             learning_rate=0.01,  # Reduced learning rate
                             depth=5,
                             reg_lambda=3,       # Added L2 regularization (reg_lambda replaces l2_leaf_reg)
                             loss_function='RMSE',
                             eval_metric='RMSE',   # Specify eval_metric for early stopping
                             verbose=0,
                             random_seed=42,
                             early_stopping_rounds=50) # Added early stopping
model_catboost.fit(X_train_catboost, y_train_catboost, eval_set=[(X_val_catboost, y_val_catboost)], verbose=0)
y_pred_catboost = model_catboost.predict(X_val_catboost)
rmse_catboost = np.sqrt(mean_squared_error(y_val_catboost, y_pred_catboost))



# --- Ensemble (Weighted Averaging based on Validation RMSE) ---
weight_lgbm = rmse_catboost / (rmse_lgbm + rmse_catboost)
weight_catboost = rmse_lgbm / (rmse_lgbm + rmse_catboost)

# --- Load and Preprocess Test Data ---
test_df = load_and_preprocess_data(os.path.join(DATA_DIR, 'test.csv'), is_train=False)


# --- Make Predictions on Test Data ---
X_test_lgbm = test_df[features_lgbm]
X_test_catboost = test_df[features_catboost]

test_pred_lgbm = model_lgbm.predict(X_test_lgbm)
test_pred_catboost = model_catboost.predict(X_test_catboost)

#Ensemble
test_pred_ensemble = (weight_lgbm * test_pred_lgbm) + (weight_catboost * test_pred_catboost)

# --- Create Submission File ---
submission = pd.DataFrame({'key': test_df['key'], 'fare_amount': test_pred_ensemble})

# Ensure the 'final' directory exists
output_dir = './final'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)

# Calculate final validation performance
y_pred_ensemble_val = (weight_lgbm * y_pred_lgbm) + (weight_catboost * y_pred_catboost)
rmse_ensemble = np.sqrt(mean_squared_error(y_val_lgbm, y_pred_ensemble_val))

print(f"Final Validation Performance: {rmse_ensemble}")
