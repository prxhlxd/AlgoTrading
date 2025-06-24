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

class AdvancedPreprocessingPipeline:
    def __init__(self):
        self.scalers = {}

    def stationarity_test(self, series):
        from statsmodels.tsa.stattools import adfuller, kpss

        # ADF Test
        adf_result = adfuller(series.dropna())
        adf_stationary = adf_result[1] < 0.05

        # KPSS Test
        kpss_result = kpss(series.dropna())
        kpss_stationary = kpss_result[1] > 0.05

        return {
            'adf_stationary': adf_stationary,
            'kpss_stationary': kpss_stationary,
            'is_stationary': adf_stationary and kpss_stationary
        }

    def make_stationary(self, df):
        # Apply differencing if needed
        for col in ['Close', 'Open', 'High', 'Low']:
            if col in df.columns:
                stationarity = self.stationarity_test(df[col])
                if not stationarity['is_stationary']:
                    df[f'{col}_diff'] = df[col].diff()
        return df

    def feature_scaling(self, X_train, X_test, method='robust'):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        for col in X_train.columns:
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            else:  # robust
                scaler = RobustScaler()

            X_train_scaled[col] = scaler.fit_transform(X_train[[col]])
            X_test_scaled[col] = scaler.transform(X_test[[col]])
            self.scalers[col] = scaler

        return X_train_scaled, X_test_scaled
