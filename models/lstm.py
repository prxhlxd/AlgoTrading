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

class LSTMPredictor:
    def __init__(self, sequence_length=60, n_features=None , feature_cols = None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.feature_cols = feature_cols
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def create_sequences(self, data):
        X, y = [], []
        # Ensure data is a pandas DataFrame or has a similar structure for target extraction
        if isinstance(data, pd.DataFrame):
            # Assuming the target column is 'Target_15min' as used in the preceding code
            target_col = 'Target_15min'
            if target_col not in data.columns:
                 raise ValueError(f"Target column '{target_col}' not found in data.")
            data_values = data[feature_cols].values # Use the defined feature columns
            target_values = data[target_col].values
        elif isinstance(data, np.ndarray):
             # If data is just the features, we need a separate target array
             # This case might need adjustment based on how you structure data input
             data_values = data
             # You'll need a corresponding y_values if data is just features
             raise TypeError("If data is a numpy array, target values must be provided separately.")
        else:
             raise TypeError("Input data must be a pandas DataFrame or numpy array.")


        for i in range(self.sequence_length, len(data_values)):
            X.append(data_values[i-self.sequence_length:i])
            y.append(target_values[i]) # Append the target value at the end of the sequence

        return np.array(X), np.array(y)


    def build_model(self):
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(128, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),

            LSTM(64, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),

            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mse']
        )

        return self.model

    def train(self, X_train_scaled, y_train_scaled, X_val_scaled=None, y_val_scaled=None, epochs=100):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]

        validation_data = (X_val_scaled, y_val_scaled) if X_val_scaled is not None else None

        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def predict(self, X_test_scaled):
        return self.model.predict(X_test_scaled)

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    def normalize_features_and_target(self, train_data, test_data):
        # Use only the feature columns for scaling X
        X_train_data = train_data[feature_cols]
        X_test_data = test_data[feature_cols]

        # Fit and transform the features
        self.scaler_X.fit(X_train_data)
        X_train_scaled_features = self.scaler_X.transform(X_train_data)
        X_test_scaled_features = self.scaler_X.transform(X_test_data)

        # Fit and transform the target
        self.scaler_y.fit(train_data['Target_15min'].values.reshape(-1, 1))
        y_train_scaled = self.scaler_y.transform(train_data['Target_15min'].values.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(test_data['Target_15min'].values.reshape(-1, 1)).flatten()

        train_data_for_scaling = train_data[feature_cols + ['Target_15min']].copy()
        test_data_for_scaling = test_data[feature_cols + ['Target_15min']].copy()

        # Scale features
        self.scaler_X.fit(train_data_for_scaling[feature_cols])
        train_data_for_scaling[feature_cols] = self.scaler_X.transform(train_data_for_scaling[feature_cols])
        test_data_for_scaling[feature_cols] = self.scaler_X.transform(test_data_for_scaling[feature_cols])

        # Scale target
        self.scaler_y.fit(train_data_for_scaling[['Target_15min']])
        train_data_for_scaling['Target_15min'] = self.scaler_y.transform(train_data_for_scaling[['Target_15min']])
        test_data_for_scaling['Target_15min'] = self.scaler_y.transform(test_data_for_scaling[['Target_15min']])

        # Now create sequences from these scaled DataFrames
        X_train_seq, y_train_seq = self.create_sequences(train_data_for_scaling)
        X_test_seq, y_test_seq = self.create_sequences(test_data_for_scaling)

        return X_train_seq, y_train_seq, X_test_seq, y_test_seq

# Example usage:
# First, ensure df_processed is available as in the preceding code
# (Assume df_processed, train_data, test_data, feature_cols are defined)

