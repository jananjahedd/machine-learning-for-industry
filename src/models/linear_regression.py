"""
File: linear_regression.py
Authors: Janan Jahed (s5107318), Andrei Medesan (s5130727),
         Andjela Matic (s5248736), Stefan Stoian (s4775309)
Description: The file trains a Linear Regression model on the training
             and dataset visualizes predictions.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent.parent.parent
MultipleDatasets = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / 'linear-regression.log'

# set up the logger
format_style = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format=format_style)


class Logger:
    """Simple logger class to log information and errors"""
    @staticmethod
    def log_info(message):
        logging.info(message)

    @staticmethod
    def log_error(message):
        logging.error(message)


class LinearRegressionModel:
    def __init__(self, regularization=None, alpha=1.0) -> None:
        """
        :param regularization: Choose between 'ridge', 'lasso',
        or None (for standard linear regression)
        :param alpha: Regularization strength; must be a positive float
        """
        self._train_df, self._val_df, self._test_df = self._load_data()
        Logger.log_info("Loading the datasets for the Linear Regression model")

        self.features = ['Avg_NO', 'Avg_NO2', 'Avg_NOX', 'Avg_O3', 'temp',
                         'humidity', 'windspeed', 'cloudcover', 'visibility',
                         'solarradiation', 'Month', 'Holiday', 'Weekday',
                         'High_O3_warm_months', 'Inverse_NO2_O3',
                         'DayOfYear', 'sin_1', 'cos_1', 'sin_2', 'cos_2']
        self.target = ['Avg_NO2', 'Avg_O3']

        self.regularization = regularization
        self.alpha = alpha
        self.model = None
        self.mse = 0.0
        self.rmse = 0.0
        self.r2 = 0.0

    def reorder_data(self) -> None:
        """
        Reorder the training, validation, and test data to prevent
        data leakage.
        """
        Logger.log_info("Reordering data to prevent data leakage...")
        try:
            all_data = pd.concat([self._train_df, self._val_df, self._test_df],
                                 axis=0)
            all_data = all_data.sort_values('datetime').reset_index(drop=True)

            total_len = len(all_data)
            train_cutoff = int(total_len * 0.6)
            val_cutoff = int(total_len * 0.8)

            buffer_size = int(total_len * 0.02)

            self._train_df = all_data.iloc[:train_cutoff
                                           ].reset_index(drop=True)
            self._val_df = all_data.iloc[train_cutoff + buffer_size:val_cutoff
                                         ].reset_index(drop=True)
            self._test_df = all_data.iloc[val_cutoff + buffer_size:
                                          ].reset_index(drop=True)

            if self._val_df.empty or self._test_df.empty:
                raise ValueError("Empty validation or test set." +
                                 "Please ensure sufficient data is available.")

            # create an unseen dataset (e.g., use last 5% of the data)
            unseen_cutoff = int(total_len * 0.95)
            self._unseen_df = all_data.iloc[unseen_cutoff:
                                            ].reset_index(drop=True)

            Logger.log_info("Data reordered successfully to prevent leakage.")
        except Exception as e:
            Logger.log_error(f"Error while reordering data: {str(e)}")

    def _load_data(self) -> MultipleDatasets:
        """Loads the datasets for usage."""
        try:
            train_df = pd.read_csv(
                root / 'data' / 'processed' / 'normalized_train_data.csv')
            val_df = pd.read_csv(
                root / 'data' / 'processed' / 'normalized_val_data.csv')
            test_df = pd.read_csv(
                root / 'data' / 'processed' / 'normalized_test_data.csv')
            Logger.log_info("Successfully loaded the datasets.")
            return train_df, val_df, test_df
        except Exception as e:
            Logger.log_error(f"Error while loading the datasets: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def check_data_leakage(self) -> bool:
        """Check for data leakage by comparing date ranges and duplicates."""
        try:
            Logger.log_info("Checking for data leakage...")

            train_dates = pd.to_datetime(self._train_df['datetime'])
            val_dates = pd.to_datetime(self._val_df['datetime'])
            test_dates = pd.to_datetime(self._test_df['datetime'])

            if val_dates.min() <= train_dates.max():
                Logger.log_error("Data leakage detected: Validation set" +
                                 "has overlapping dates with the training")
                return True

            if test_dates.min() <= val_dates.max():
                Logger.log_error("Data leakage detected: Test set" +
                                 "has overlapping dates with the validation.")
                return True

            duplicates_train_val = pd.merge(self._train_df,
                                            self._val_df, how='inner')
            duplicates_train_test = pd.merge(self._train_df,
                                             self._test_df, how='inner')
            duplicates_val_test = pd.merge(self._val_df,
                                           self._test_df, how='inner')

            if not duplicates_train_val.empty:
                Logger.log_error(f"Leakage found:{len(duplicates_train_val)}" +
                                 "duplicate rows between training and val")
                return True

            if not duplicates_train_test.empty:
                Logger.log_error(f"Leakage:{len(duplicates_train_test)}" +
                                 "duplicate rows between training and test")
                return True

            if not duplicates_val_test.empty:
                Logger.log_error(f"leakage: {len(duplicates_val_test)}" +
                                 "duplicate rows between validation and test.")
                return True

            Logger.log_info("No data leakage detected.")
            return False

        except Exception as e:
            Logger.log_error("Error while checking for data leakage:" +
                             f"{str(e)}")
            return True

    def save_metrics_to_json(self, metrics: dict) -> None:
        """
        Save metrics to a JSON file without overwriting existing data.
        :param metrics: dictionary containing all metrics
        """
        file_path = Path('results')
        file_path.mkdir(parents=True, exist_ok=True)
        filename = file_path / 'linear_regression_metrics.json'

        # Load existing data if the file already exists
        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}  # Start fresh if file is corrupted
        else:
            existing_data = {}

        # Update existing data with new metrics
        existing_data.update(metrics)

        # Save the updated metrics
        try:
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=4)
        except Exception as e:
            Logger.log_error(f"Error while saving metrics to JSON: {str(e)}")

    def train_model(self, n_splits: int = 5) -> None:
        """
        Train the Linear Regression model with K-Fold cross-validation.

        :param: n_splits: the number of splits in K-Fold.
        """
        try:
            Logger.log_info("Training the Linear Regression model " +
                            "with K-Fold cross validation...")
            X = (
                self._train_df[self.features]
                .apply(pd.to_numeric, errors='coerce').dropna()
            )
            y = (
                self._train_df[self.target]
                .apply(pd.to_numeric, errors='coerce')
                .dropna()
            )

            valid_indices = X.dropna().index.intersection(y.dropna().index)
            X = X.loc[valid_indices]
            y = y.loc[valid_indices]

            if self.regularization == 'ridge':
                self.model = Ridge(alpha=self.alpha)
                Logger.log_info(f"L2 regularization with alpha={self.alpha}.")
            elif self.regularization == 'lasso':
                self.model = Lasso(alpha=self.alpha)
                Logger.log_info(f"L1 regularization with alpha={self.alpha}.")
            else:
                self.model = LinearRegression()
                Logger.log_info("Using Standard Linear Regression.")

            # K-Fold cross-validation
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            mse_scores, rmse_scores, r2_scores = [], [], []

            for train_index, val_index in kf.split(X):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                self.model.fit(X_train, y_train)
                predictions = self.model.predict(X_val)

                mse = np.mean((y_val - predictions) ** 2)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_val, predictions)

                mse_scores.append(mse)
                rmse_scores.append(rmse)
                r2_scores.append(r2)

                Logger.log_info(f"Fold MSE: {mse:.4f}, RMSE: " +
                                f"{rmse:.4f}, R2: {r2:.4f}")

            # Average scores across all folds
            self.mse = np.mean(mse_scores)
            self.rmse = np.mean(rmse_scores)
            self.r2 = np.mean(r2_scores)

            Logger.log_info("Cross-Validation Results - MSE: " +
                            f"{self.mse:.4f}, RMSE: {self.rmse:.4f}, " +
                            f"R2: {self.r2:.4f}")
            print(f"Training - MSE: {self.mse:.4f}, RMSE: {self.rmse:.4f}, " +
                  f"R2: {self.r2:.4f}")

            # save metrics
            metrics = {"Cross-Validation": {"MSE": self.mse, "RMSE": self.rmse,
                                            "R2": self.r2}}
            self.save_metrics_to_json(metrics)

        except Exception as e:
            Logger.log_error(f"Error during training with K-Fold CV: {str(e)}")

    def evaluate_on_test(self):
        """Evaluate model on the test set to check for overfitting."""
        try:
            Logger.log_info("Evaluating model on the test dataset...")
            X_test = (self._test_df[self.features]
                      .apply(pd.to_numeric, errors='coerce')
                      .dropna())
            y_test = (self._test_df[self.target]
                      .apply(pd.to_numeric, errors='coerce')
                      .dropna())

            test_predictions = self.model.predict(X_test)

            test_mse = np.mean((y_test - test_predictions) ** 2)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(y_test, test_predictions)

            Logger.log_info(f"Test MSE: {test_mse:.4f}, " +
                            f"Test RMSE: {test_rmse:.4f}, " +
                            f"Test R2: {test_r2:.4f}")
            print(f"Test - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, " +
                  f"R2: {test_r2:.4f}")

            metrics = {"Test": {"MSE": test_mse, "RMSE": test_rmse,
                                "R2": test_r2}}
            self.save_metrics_to_json(metrics)

            # check for overfitting
            self.check_overfitting(test_r2)

        except Exception as e:
            Logger.log_error(f"Error during evaluation on test set: {str(e)}")

    def check_overfitting(self, test_r2: float) -> None:
        """
        Check if overfitting is occurring by comparing
        training and test R2 scores.

        :param: test_r2: float to calculate presence of
                         overfitting.
        """
        try:
            Logger.log_info("Checking for overfitting...")

            if abs(self.r2 - test_r2) > 0.1:  # Threshold to detect overfitting
                Logger.log_error("Overfitting detected: " +
                                 f"Training R2 = {self.r2:.4f}, " +
                                 f"Test R2 = {test_r2:.4f}")
                Logger.log_info("Reducing model complexity to prevent" +
                                "overfitting")
                # Reduce model complexity (e.g., increase regularization)
                if self.regularization == 'ridge':
                    self.alpha *= 2
                    Logger.log_info(f"Increasing Ridge alpha to {self.alpha}.")
                    self.train_model()
                    self.evaluate_on_test()
                elif self.regularization == 'lasso':
                    self.alpha *= 2
                    Logger.log_info(f"Increasing Lasso alpha to {self.alpha}.")
                    self.train_model()
                    self.evaluate_on_test()
                else:
                    Logger.log_info("Prevent overfitting.")
            else:
                Logger.log_info("No significant overfitting detected.")
        except Exception as e:
            Logger.log_error(f"Error while checking for overfitting: {str(e)}")

    def evaluate_on_unseen_data(self) -> None:
        """Evaluate the model on unseen data."""
        try:
            Logger.log_info("Evaluating model on unseen dataset...")
            X_unseen = (self._unseen_df[self.features]
                        .apply(pd.to_numeric, errors='coerce')
                        .dropna())
            y_unseen = (self._unseen_df[self.target]
                        .apply(pd.to_numeric, errors='coerce')
                        .dropna())

            unseen_predictions = self.model.predict(X_unseen)

            unseen_mse = np.mean((y_unseen - unseen_predictions) ** 2)
            unseen_rmse = np.sqrt(unseen_mse)
            unseen_r2 = r2_score(y_unseen, unseen_predictions)

            Logger.log_info(f"Unseen Data MSE: {unseen_mse:.4f}, "
                            f"RMSE: {unseen_rmse:.4f}, R2: {unseen_r2:.4f}")
            print(f"Unseen Data - MSE: {unseen_mse:.4f}, " +
                  f"RMSE: {unseen_rmse:.4f}, R2: {unseen_r2:.4f}")

            metrics = {"Unseen Data": {"MSE": unseen_mse, "RMSE": unseen_rmse,
                                       "R2": unseen_r2}}
            self.save_metrics_to_json(metrics)

        except Exception as e:
            Logger.log_error(f"Error while evaluating unseen data: {str(e)}")

    def predict(self) -> None:
        """Generate predictions and save them in results/ for dashboard."""
        try:
            Logger.log_info("Generating predictions...")

            X_test = (self._test_df[self.features]
                      .apply(pd.to_numeric, errors='coerce')
                      .dropna())
            predictions = self.model.predict(X_test)

            results_dir = root / 'results' / 'predictions'
            results_dir.mkdir(parents=True, exist_ok=True)

            results_path = results_dir / 'linear_regression_predictions.csv'
            np.savetxt(results_path, predictions, delimiter=',')
            Logger.log_info(f"Predictions saved to file: {results_path}")

            # After saving predictions, plot them
            self.plot_predictions(predictions)

        except Exception as e:
            Logger.log_error(f"Error while generating predictions: {str(e)}")
            return None

    def plot_predictions(self, predictions: List[float]) -> None:
        """
        Plot NO2 and O3 predictions against actual values.

        :param predictions: list of predictions for plotting
        """
        try:
            Logger.log_info("Plotting Linear Regression predictions...")

            # Get the actual test values
            y_test = (self._test_df[self.target]
                      .apply(pd.to_numeric, errors='coerce')
                      .dropna())

            if predictions is None or y_test.empty:
                raise ValueError("No predictions or test data found.")

            plt.figure(figsize=(12, 6))

            # plot for NO2
            plt.subplot(1, 2, 1)
            plt.plot(y_test['Avg_NO2'].values, label="Actual NO2")
            plt.plot(predictions[:, 0], label="Predicted NO2", linestyle='--')
            plt.title("Linear Regression - NO2 Predictions")
            plt.legend()

            # plot for O3
            plt.subplot(1, 2, 2)
            plt.plot(y_test['Avg_O3'].values, label="Actual O3")
            plt.plot(predictions[:, 1], label="Predicted O3", linestyle='--')
            plt.title("Linear Regression - O3 Predictions")
            plt.legend()

            plt.tight_layout()

            # save the plot to the results folder
            plot_dir = root / 'results' / 'plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / 'linear_regression_plot.png'
            plt.savefig(plot_path)
            plt.close()

            Logger.log_info(f"Plot saved for Linear Regression: {plot_path}")

        except Exception as e:
            Logger.log_error(f"Error while plotting predictions: {str(e)}")


def main():
    # You can choose regularization as 'ridge', 'lasso', or
    # None (standard linear regression)
    linear_model = LinearRegressionModel(regularization='ridge',
                                         alpha=1.0)

    linear_model.reorder_data()

    if not linear_model.check_data_leakage():
        linear_model.train_model()
        linear_model.evaluate_on_test()
        linear_model.predict()
        linear_model.evaluate_on_unseen_data()
