"""
File: random_forest.py
Authors: Janan Jahed (s5107318), Andrei Medesan (s5130727),
         Andjela Matic (s5248736), Stefan Stoian (s4775309)
Description: The file trains a Random Forest Regressor on the
             training dataset and visualizes predictions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent.parent
MultipleDatasets = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / 'random_forest.log'

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


class RandomForestModel:
    def __init__(self) -> None:
        """Initializes the Random Forest class."""
        Logger.log_info("Loading the datasets for the Random Forest model...")
        self._train_df, self._val_df, self._test_df = self._load_data()

        # Updated features: Exclude target variables from the features list
        self.features = ['Avg_NO', 'Avg_NOX', 'temp', 'humidity', 'cloudcover',
                         'visibility', 'solarradiation', 'Month', 'Weekday',
                         'Holiday', 'Inverse_NO2_O3', 'High_O3_warm_months']

        self.target = ['Avg_NO2', 'Avg_O3']
        self.model = RandomForestRegressor()

        # metrics to obtain from the training dataset
        self.mse = 0.0
        self.rmse = 0.0
        self.r2 = 0.0

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

    def _save_best_params(self, best_params: Dict) -> None:
        """
        Saves the best parameters after performing grid search
        :param best_params: dictionary containing the parameters
                            found in grid search.
        """
        params_dir = root / 'src' / 'models'
        params_dir.mkdir(parents=True, exist_ok=True)
        params_path = params_dir / 'random_forest_params.json'

        with open(params_path, 'w') as f:
            json.dump(best_params, f)
        Logger.log_info("The parameters were saved as " +
                        f"JSON file at {params_path}")

    def _load_best_params(self) -> Optional[Dict]:
        """Load the best parameters from the JSON file."""
        params_path = root / 'src' / 'models' / 'random_forest_params.json'
        if params_path.exists():
            with open(params_path, 'r') as f:
                best_params = json.load(f)
            Logger.log_info("The parameters have been loaded" +
                            f" from the file {params_path}")
            return best_params
        Logger.log_info("No parameters were found in the file." +
                        " Please check or perform grid search.")
        return None

    def grid_search(self, x_train: pd.DataFrame,
                    y_train: pd.DataFrame) -> None:
        """
        Perform grid search to find the best hyperparameters.
        :param x_train: the independent training data
        :param y_train: the dependent training data
        """
        gr_space = {
            'max_depth': [3, 5, 7, 10],
            'n_estimators': [100, 200, 300, 400, 500],
            'max_features': [10, 20, 30, 40, 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'random_state': [0, 42],
            'min_samples_split': [2, 5]
        }

        grid = GridSearchCV(estimator=self.model, param_grid=gr_space, cv=3,
                            scoring='neg_mean_squared_error', verbose=3)
        # Fit the grid search
        grid.fit(x_train, y_train)

        # Set the best estimator found by grid search as the model
        self.model = grid.best_estimator_
        Logger.log_info(f"Best parameters found: {grid.best_params_}")

        # save the best parameters found
        self._save_best_params(grid.best_params_)

    def check_for_leakage(self) -> None:
        """Check for data leakage in the features."""
        leakage_features = [feature for feature in self.features
                            if feature in self.target]
        if leakage_features:
            Logger.log_error(f"Leakage detected: {leakage_features}")
            raise ValueError("Leakage found: target " +
                             "variable included in features.")
        else:
            Logger.log_info("No target leakage detected.")

    def feature_importance(self) -> None:
        """Log feature importance after training."""
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        Logger.log_info(f"Feature importances:\n{feature_importance_df}")
        print(feature_importance_df)

    def train(self) -> None:
        """Train the Random Forest model using saved hyperparameters."""
        try:
            Logger.log_info("Training the Random Forest model...")

            # Ensure only numeric columns are used and handle NaN values
            X_train = (self._train_df[self.features]
                       .apply(pd.to_numeric, errors='coerce')
                       .dropna())
            Y_train = (self._train_df[self.target]
                       .apply(pd.to_numeric, errors='coerce')
                       .dropna())

            valid_indices = X_train.dropna().index.intersection(
                Y_train.dropna().index)
            X_train = X_train.loc[valid_indices]
            Y_train = Y_train.loc[valid_indices]

            # Check for data leakage before training
            self.check_for_leakage()

            best_params = self._load_best_params()
            if best_params:
                self.model.set_params(**best_params)
                Logger.log_info(f"Using saved best parameters: {best_params}")
            else:
                self.grid_search(X_train, Y_train)

            # K-Fold Cross Validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mse_scores = []

            # store the losses
            train_loss = []
            val_loss = []

            for train_index, val_index in kf.split(X_train):
                X_kf_train, X_kf_val = (
                    X_train.iloc[train_index], X_train.iloc[val_index]
                )
                Y_kf_train, Y_kf_val = (
                    Y_train.iloc[train_index], Y_train.iloc[val_index]
                )

                # Fit the model on the training fold
                self.model.fit(X_kf_train, Y_kf_train)
                train_pred = self.model.predict(X_kf_train)
                val_pred = self.model.predict(X_kf_val)

                # Calculate the MSE for training and validation sets
                train_mse = np.mean((Y_kf_train - train_pred) ** 2)
                val_mse = np.mean((Y_kf_val - val_pred) ** 2)

                train_loss.append(train_mse)
                val_loss.append(val_mse)

                predictions = self.model.predict(X_kf_val)

                # Calculate and store the MSE for this fold
                mse = np.mean((Y_kf_val - predictions) ** 2)
                mse_scores.append(mse)

            mean_mse = np.mean(mse_scores)
            Logger.log_info(f"K-Fold MSE scores: {mse_scores}")
            Logger.log_info(f"Mean K-Fold Cross-validation MSE: {mean_mse}")

            # Fit the model on the entire training set after cross-validation
            self.model.fit(X_train, Y_train)
            Logger.log_info("Model training completed.")

            predictions = self.model.predict(X_train)

            self.mse = np.mean((Y_train - predictions) ** 2)
            self.rmse = np.sqrt(self.mse)
            self.r2 = r2_score(Y_train, predictions)
            self._plot_losses(train_loss, val_loss)

            self.feature_importance()

            Logger.log_info(f"Training MSE: {self.mse:.4f}, " +
                            f"RMSE: {self.rmse:.4f}, R2: {self.r2:.4f}")
            print(f"Training - MSE: {self.mse:.4f}, " +
                  f"RMSE: {self.rmse:.4f}, R2: {self.r2:.4f}")

        except Exception as e:
            Logger.log_error(f"Error during training: {str(e)}")

    def _plot_losses(self, train_losses: list, val_losses: list) -> None:
        """
        Plot training and validation losses.
        :param train_losses: list of train losses
        :param val_losses: list of val losses
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", marker='o', color='blue')
        plt.plot(val_losses, label="Validation Loss", marker='o', color='red')
        plt.title("Training and Validation Loss per Fold")
        plt.xlabel("Fold")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)

        # save the plot in the designated directory
        plot_dir = root / 'results' / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / 'RF_loss.png'
        plt.savefig(plot_path)
        plt.close()

    def predict(self) -> None:
        """Generate predictions and save them in results/ for dashboard."""
        try:
            Logger.log_info("Generating predictions...")

            X_test = (self._test_df[self.features]
                      .apply(pd.to_numeric, errors='coerce')
                      .dropna())
            y_test = (self._test_df[self.target]
                      .apply(pd.to_numeric, errors='coerce')
                      .dropna())

            valid_indices = X_test.dropna().index.intersection(
                y_test.dropna().index)
            X_test = X_test.loc[valid_indices]
            y_test = y_test.loc[valid_indices]

            test_predictions = self.model.predict(X_test)

            test_mse = np.mean((y_test - test_predictions) ** 2)
            test_rmse = np.sqrt(test_mse)
            test_r2 = r2_score(y_test, test_predictions)

            Logger.log_info(f"Test MSE: {test_mse:.4f}, " +
                            f"Test RMSE: {test_rmse:.4f}, " +
                            f"Test R2: {test_r2:.4f}")
            print(f"Test - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, " +
                  f"R2: {test_r2:.4f}")

            results_dir = root / 'results' / 'predictions'
            results_dir.mkdir(parents=True, exist_ok=True)

            results_path = results_dir / 'random_forest_predictions.csv'
            np.savetxt(results_path, test_predictions, delimiter=',')
            Logger.log_info(f"Predictions saved to file: {results_path}")

            self.plot_predictions(test_predictions)

        except Exception as e:
            Logger.log_error(f"Error while generating predictions: {str(e)}")
            return None

    def plot_predictions(self, predictions: List[float]) -> None:
        """
        Plot NO2 and O3 predictions against actual values.
        :param predictions: list of predictions
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
            plt.title("Random Forest - NO2 Predictions")
            plt.legend()

            # plot for O3
            plt.subplot(1, 2, 2)
            plt.plot(y_test['Avg_O3'].values, label="Actual O3")
            plt.plot(predictions[:, 1], label="Predicted O3", linestyle='--')
            plt.title("Random Forest - O3 Predictions")
            plt.legend()

            plt.tight_layout()

            # save the plot to the results folder
            plot_dir = root / 'results' / 'plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / 'random_forest_plot.png'
            plt.savefig(plot_path)
            plt.close()

            Logger.log_info(f"Plot saved for Random Forest: {plot_path}")

        except Exception as e:
            Logger.log_error(f"Error while plotting predictions: {str(e)}")


if __name__ == "__main__":
    random_forest_instance = RandomForestModel()
    random_forest_instance.train()
    random_forest_instance.predict()
