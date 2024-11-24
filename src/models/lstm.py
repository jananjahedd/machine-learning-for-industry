"""
File: lstm.py
Authors: Janan Jahed (s5107318), Andrei Medesan (s5130727),
         Andjela Matic (s5248736), Stefan Stoian (s4775309)
Description: The file trains the LSTM model and evaluates.
"""
from pathlib import Path
from typing import Tuple
import logging
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Layer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import tensorflow.keras.backend as K


# Define the paths and setup the log file
root = Path(__file__).resolve().parent.parent.parent
MultipleDatasets = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / 'lstm.log'

# Set up the logger
format_style = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    filename=log_file_path, level=logging.INFO, format=format_style
)


class Logger:
    """Simple logger class to log information and errors."""

    @staticmethod
    def log_info(message):
        """Logs informational messages."""
        logging.info(message)

    @staticmethod
    def log_error(message):
        """Logs error messages."""
        logging.error(message)


class Attention(Layer):
    """Custom attention mechanism."""

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """Creates attention layer weights."""
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform", trainable=True
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[-1],),
            initializer="zeros", trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        """Applies attention weights to the input."""
        et = K.tanh(K.dot(x, self.W) + self.b)
        at = K.softmax(et, axis=1)
        output = x * at
        return K.sum(output, axis=1)


class LSTMModel:
    """LSTM model for time series forecasting with attention."""

    def __init__(self, sequence_length: int = 14) -> None:
        """Initializes the LSTM model and loads datasets."""
        self._train_df, self._val_df, self._test_df = self._load_data()
        self.seq_length = sequence_length
        self._model = None
        self.history = None
        Logger.log_info("LSTM model initialized and datasets were loaded.")

    def _load_data(self) -> MultipleDatasets:
        """Loads training, validation, and test data."""
        try:
            train_df = pd.read_csv(
                root / 'data' / 'splits' / 'train_data.csv'
            )
            val_df = pd.read_csv(
                root / 'data' / 'splits' / 'val_data.csv'
            )
            test_df = pd.read_csv(
                root / 'data' / 'splits' / 'test_data.csv'
            )

            # Add rolling averages for NO2 and O3 to smooth noise
            train_df['NO2_week_avg'] = (
                train_df['Avg_NO2'].rolling(window=7).mean().bfill()
            )
            train_df['O3_week_avg'] = (
                train_df['Avg_O3'].rolling(window=7).mean().bfill()
            )
            val_df['NO2_week_avg'] = (
                val_df['Avg_NO2'].rolling(window=7).mean().bfill()
            )
            val_df['O3_week_avg'] = (
                val_df['Avg_O3'].rolling(window=7).mean().bfill()
            )
            test_df['NO2_week_avg'] = (
                test_df['Avg_NO2'].rolling(window=7).mean().bfill()
            )
            test_df['O3_week_avg'] = (
                test_df['Avg_O3'].rolling(window=7).mean().bfill()
            )

            Logger.log_info("Successfully loaded the datasets.")
            return train_df, val_df, test_df
        except Exception as e:
            Logger.log_error(f"Error loading datasets: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def preprocess_data(self) -> None:
        """Preprocesses the data for LSTM and creates sequences."""
        self._train_df = (self._train_df.fillna(method='bfill')
                          .fillna(method='ffill'))
        self._val_df = (self._val_df.fillna(method='bfill')
                        .fillna(method='ffill'))
        self._test_df = (self._test_df.fillna(method='bfill')
                         .fillna(method='ffill'))

        def create_seq(data: pd.DataFrame, seq_length: int,
                       target: str) -> Tuple[np.ndarray, np.ndarray]:
            """Creates sequences from the data."""
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length, :-2])
                y.append(data[i + seq_length, target])
            return np.array(X), np.array(y)

        # Filter for only numeric data points
        numeric_train_df = self._train_df.select_dtypes(include=[np.number])
        numeric_val_df = self._val_df.select_dtypes(include=[np.number])
        numeric_test_df = self._test_df.select_dtypes(include=[np.number])

        # Create the sequences for O3
        self.X_train_o3, self.y_train_o3 = create_seq(
            numeric_train_df.values, self.seq_length, -1  # O3 column index
        )
        self.X_val_o3, self.y_val_o3 = create_seq(
            numeric_val_df.values, self.seq_length, -1
        )
        self.X_test_o3, self.y_test_o3 = create_seq(
            numeric_test_df.values, self.seq_length, -1
        )

        # Create the sequences for NO2
        self.X_train_no2, self.y_train_no2 = create_seq(
            numeric_train_df.values, self.seq_length, -2  # NO2 column index
        )
        self.X_val_no2, self.y_val_no2 = create_seq(
            numeric_val_df.values, self.seq_length, -2
        )
        self.X_test_no2, self.y_test_no2 = create_seq(
            numeric_test_df.values, self.seq_length, -2
        )

        Logger.log_info("Data preprocessed for both O3 and NO2 models.")

    def build_model(self, hp: kt.HyperParameters):
        """Builds the LSTM model using hyperparameters from Keras Tuner."""
        model = Sequential()
        # Hyperparameter choices from the tuner
        units = hp.Int('units', min_value=64, max_value=256, step=32)
        dropout_rate = hp.Choice('dropout_rate', values=[0.2, 0.3, 0.4])
        learning_rate = hp.Choice('learning_rate',
                                  values=[0.001, 0.0001, 0.00001])
        num_layers = hp.Int('num_layers', min_value=1, max_value=3)

        for _ in range(num_layers):
            model.add(Bidirectional(
                LSTM(units=units, return_sequences=True,
                     kernel_regularizer=l2(0.01))
            ))
            model.add(Dropout(dropout_rate))

        model.add(Dense(units=256, activation='tanh'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=128, activation='tanh'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=64, activation='tanh'))
        model.add(Dropout(dropout_rate))
        # Adding Attention Layer after LSTM layers
        model.add(Attention())

        # Output layer
        model.add(Dense(1))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error',
                      metrics=['mae'])

        return model

    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           target_name: str) -> None:
        """Trains and evaluates a model for a specific target (O3 or NO2)."""
        tuner = kt.RandomSearch(
            self.build_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=2,
            directory='kt_logs',
            project_name=f'{target_name}_lstm_hyperparameter_tuning'
        )

        # define early stopping and run
        tuner.search(
            X_train, y_train, epochs=100,
            validation_data=(X_val, y_val),
            callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
        )

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=6)[0]
        Logger.log_info("Best hyperparameters for " +
                        f"{target_name}: {best_hps.values}")

        # Build the model with the best hyperparameters
        best_model = tuner.hypermodel.build(best_hps)

        # Train the best model
        self.train_model(best_model, X_train, y_train, X_val, y_val,
                         target_name, epochs=100)

        # Evaluate the model
        self.evaluate_model(best_model, X_test, y_test, target_name)

    def train_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray, target_name: str,
                    epochs: int = 100, batch_size: int = 64) -> None:
        """Trains the LSTM model and plots training and validation loss."""
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        try:
            self.history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs, batch_size=batch_size,
                callbacks=[early_stopping]
            )
            Logger.log_info(f"{target_name} Model training " +
                            f"completed with {epochs} epochs.")

            # Plot the training and validation loss
            self.plot_loss(self.history, target_name)

        except Exception as e:
            Logger.log_error(f"Error during {target_name} " +
                             f"model training: {str(e)}")

    def plot_loss(self, history, target_name: str) -> None:
        """Plots the training and validation loss."""
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{target_name} Model - Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # save the plot for the dashboard
        plot_dir = root / 'results' / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f'lstm_{target_name}_loss.png'
        plt.savefig(plot_path)
        plt.close()

    def evaluate_model(self, model, X_test: np.ndarray,
                       y_test: np.ndarray, target_name: str) -> None:
        """Evaluates the trained model on the test
        data and plots predictions."""
        try:
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            Logger.log_info(f"{target_name} - Mean Absolute Error: {mae}")
            Logger.log_info(f"{target_name} - Root Mean Squared Error: {rmse}")
            Logger.log_info(f"{target_name} - R² Score: {r2}")
            print(f"{target_name} - Mean Absolute Error (MAE): {mae}")
            print(f"{target_name} - Root Mean Squared Error (RMSE): {rmse}")
            print(f"{target_name} - R² Score: {r2}")

            metrics = {'test_mae': mae, 'test_rmse': rmse, 'test_r2': r2}
            metrics_file = f'lstm_{target_name}_metrics.json'
            metrics_path = root / 'results' / metrics_file
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)

            # Plot predicted vs. actual values
            self.plot_predictions(y_test, y_pred, target_name)

        except Exception as e:
            Logger.log_error(f"Error during {target_name} " +
                             f"model evaluation: {str(e)}")

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                         target_name: str) -> None:
        """Plots predicted vs actual values."""
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.title(f'{target_name} Model - Predicted vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                 'r--')

        # save the plot for the dashboard
        plot_dir = root / 'results' / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f'lstm_{target_name}_predictions.png'
        plt.savefig(plot_path)
        plt.close()


if __name__ == "__main__":
    lstm = LSTMModel()
    lstm.preprocess_data()

    # Train and evaluate for O3
    lstm.train_and_evaluate(lstm.X_train_o3, lstm.y_train_o3,
                            lstm.X_val_o3, lstm.y_val_o3, lstm.X_test_o3,
                            lstm.y_test_o3, 'O3')

    # Train and evaluate for NO2
    lstm.train_and_evaluate(lstm.X_train_no2, lstm.y_train_no2,
                            lstm.X_val_no2, lstm.y_val_no2, lstm.X_test_no2,
                            lstm.y_test_no2, 'NO2')
