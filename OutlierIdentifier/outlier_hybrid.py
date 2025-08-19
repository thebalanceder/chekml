import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

class OutlierHybrid:
    def __init__(self, contamination=0.1, encoding_dim=10, epochs=50, batch_size=32):
        self.contamination = contamination
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.if_model = IsolationForest(contamination=contamination, random_state=42)
        self.lof_model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        
    def _train_autoencoder(self, X):
        input_dim = X.shape[1]
        model = Sequential([
            Dense(self.encoding_dim, activation='relu', input_shape=(input_dim,)),
            Dense(input_dim, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return model
    
    def _hybrid_cnn_rnn(self, X):
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
            Flatten(),
            LSTM(self.encoding_dim, return_sequences=False),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def detect_outliers(self, X):
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        if_scores = -self.if_model.fit(X_scaled).decision_function(X_scaled)
        
        # Local Outlier Factor
        lof_scores = -self.lof_model.fit_predict(X_scaled)
        lof_scores[lof_scores == 1] = 0
        
        # Autoencoder
        autoencoder = self._train_autoencoder(X_scaled)
        X_reconstructed = autoencoder.predict(X_scaled, verbose=0)
        autoencoder_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
        
        # Combine scores
        final_scores = (0.5 * if_scores) + (0.3 * lof_scores) + (0.2 * autoencoder_errors)
        threshold = np.percentile(final_scores, 100 * (1 - self.contamination))
        anomalies = final_scores > threshold
        
        return anomalies, final_scores
    
    def hybrid_regression(self, X_train, y_train, X_test):
        models = [LinearRegression(), DecisionTreeRegressor()]
        predictions = []
        for model in models:
            model.fit(X_train, y_train)
            predictions.append(model.predict(X_test))
        return np.mean(predictions, axis=0)
