# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os


class StudentScoreModelTrainer:
    def __init__(self, data_path='data/exams.csv'):
        self.data_path = data_path
        self.categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
        self.label_encoders = {}
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None

    def load_and_preprocess_data(self):
        data = pd.read_csv(self.data_path)

        # Encode categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.label_encoders[col] = le

        # Features and Target
        X = data[['reading score', 'writing score'] + self.categorical_cols].values
        y = data['math score'].values.reshape(-1, 1)

        # Scale
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    def build_model(self, input_dim):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        self.model = model

    def train_model(self, save_path='model.h5'):
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        self.build_model(X_train.shape[1])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        history = self.model.fit(X_train, y_train, epochs=100, batch_size=32,
                                 validation_split=0.2, callbacks=[early_stopping], verbose=1)

        # Plot training history
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

        # Evaluate
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_test_orig = self.scaler_y.inverse_transform(y_test)

        mse = mean_squared_error(y_test_orig, y_pred)
        mae = mean_absolute_error(y_test_orig, y_pred)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")

        # Scatter plot
        plt.scatter(y_test_orig, y_pred, alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')
        plt.xlabel('Actual Math Score')
        plt.ylabel('Predicted Math Score')
        plt.title('Actual vs Predicted Math Scores')
        plt.show()

        # Save model and tools
        self.model.save(save_path)
        joblib.dump(self.scaler_X, 'scaler_X.pkl')
        joblib.dump(self.scaler_y, 'scaler_y.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        print("✅ Model and preprocessing tools saved.")


# For direct script execution
if __name__ == "__main__":
    trainer = StudentScoreModelTrainer(data_path='data/exams.csv')
    trainer.train_model()
