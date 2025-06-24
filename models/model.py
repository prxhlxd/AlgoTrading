import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from scipy import signal
from scipy.stats import boxcox
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

class TreeBasedPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose = 1
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Trainig Tree")
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importance

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_with_uncertainty(self, X_test):
        # Get predictions from all trees
        predictions = np.array([tree.predict(X_test) for tree in self.model.estimators_])
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)


class GradientBoostingPredictor:
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',  # Explicitly set objective for regression
            n_estimators=2000,  # Increased estimators
            max_depth=10,       # Increased depth
            learning_rate=0.005, # Decreased learning rate
            subsample=0.9,      # Adjusted subsample
            colsample_bytree=0.9, # Adjusted colsample
            reg_alpha=0.2,      # Adjusted L1 regularization
            reg_lambda=0.2,      # Adjusted L2 regularization
            gamma=0.1,          # Added minimum loss reduction
            min_child_weight=1, # Added minimum sum of instance weight needed in a child
            random_state=42,
            early_stopping_rounds=75, # Adjusted early stopping
            eval_metric='rmse'
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=1
            )
        else:
            self.model.fit(X_train, y_train)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importance

    def predict(self, X_test):
        return self.model.predict(X_test)
    

