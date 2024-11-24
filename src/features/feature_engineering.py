"""
File: feature_engineering.py
Things to do:
- determine which function to keep for normalization
- finish feature transformation
- complete main function below
"""
import logging
from pathlib import Path
from typing import Tuple
import holidays
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Define the path to the file
root = Path(__file__).resolve().parent.parent.parent

# create typing hint
MultipleDatasets = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

# set the log directory for information to be stored
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / 'feature-engineering.log'

format_style = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format=format_style)


class Logger:
    """Simple logger file to store the information"""
    @staticmethod
    def log_info(message):
        logging.info(message)

    @staticmethod
    def log_error(message):
        logging.error(message)

    @staticmethod
    def log_warning(message):
        logging.warning(message)


class DataLoader:
    """Loads and processes datasets from the specified file paths."""
    def __init__(self, file_path: Path) -> None:
        """
        Loads the data.
        :param file_path: the path to the data.
        """
        self.file_path = file_path
        Logger.log_info(f"Loading the dataset from {self.file_path}...")

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from the file path."""
        try:
            data = pd.read_csv(self.file_path)
            Logger.log_info("Successfully loaded the data.")
            return data
        except Exception as e:
            Logger.log_error(f"Error while loading the data: {str(e)}")
            return pd.DataFrame()


class LabellingStrategy:
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Labels the data by adding more columns to help in model training.
        :param data: the data frame for labelling.
        """
        self._df = data
        Logger.log_info("Initializing labelling of the processed data...")

        self._df['datetime'] = pd.to_datetime(
            self._df['datetime'], errors='coerce'
        )

        # check if there are any invalid datetime entries
        if self._df['datetime'].isna().sum() > 0:
            Logger.log_warning("There are " +
                               f"{self._df['datetime'].isna().sum()}" +
                               " invalid datetime entries.")
            # remove the invalid rows
            self._df.dropna(subset=['datetime'], inplace=True)

    def _extract_months(self) -> None:
        """Adds names of the month."""
        Logger.log_info("Extracting month names for the data...")
        self._df['Date'] = pd.to_datetime(self._df['datetime'],
                                          utc=False)
        self._df['Month'] = self._df['Date'].dt.month_name()

    def _add_holidays(self) -> None:
        """Adds the Dutch holidays."""
        Logger.log_info("Adding holidays to the data...")
        NL_holidays = holidays.Netherlands(
            years=self._df['datetime'].dt.year.unique().tolist()
        )
        self._df['Holiday'] = self._df['datetime'].dt.date.apply(
            lambda x: x in NL_holidays
        )

    def _add_weekdays(self) -> None:
        """Adds the names of the week"""
        Logger.log_info("Adding weekday names to the data...")
        self._df['Weekday'] = self._df['datetime'].dt.day_name()

    def _add_seasonal_labels(self) -> None:
        """Adds labels for high O3 in warm months."""
        Logger.log_info("Adding labels for high O3" +
                        " levels during warmer months.")
        warm_months = ['May', 'June', 'July', 'August', 'September']
        self._df['High_O3_warm_months'] = self._df.apply(
            lambda row: row['Avg_O3'] > 80 and row['Month'] in warm_months,
            axis=1
        )

    def _add_inverse_relationship(self) -> None:
        """Adding inverse flag for NO2 and O3 and the difference."""
        Logger.log_info("Adding flag for the inverse relationship " +
                        "between NO2 and O3.")
        self._df['Inverse_NO2_O3'] = self._df.apply(
            lambda row: 1 if (row['Avg_NO2'] > 20 and row['Avg_O3'] < 39.98) or
            (row['Avg_NO2'] < 10.23 and row['Avg_O3'] > 63.39) else 0, axis=1
        )
        # add a relative difference feature between NO2 and O3
        Logger.log_info("Adding relative difference between the targets.")
        self._df['Relative_Difference'] = (
            (self._df['Avg_NO2'] - self._df['Avg_O3']).abs()
        )

    def _add_fourier_features(self, K: int = 2):
        """Adds fourier values for periodicity."""
        Logger.log_info("Adding Fourier series features" +
                        " for seasonal patterns in NO2 and O3...")
        T = 365

        self._df['DayOfYear'] = self._df['datetime'].dt.dayofyear

        for k in range(1, K + 1):
            self._df[f'sin_{k}'] = np.sin(
                2 * np.pi * k * self._df['DayOfYear'] / T
            )
            self._df[f'cos_{k}'] = np.cos(
                2 * np.pi * k * self._df['DayOfYear'] / T
            )

        Logger.log_info(f"Fourier series features added with {K} harmonics.")

    def _run(self) -> None:
        """Pipeline for Labelling Strategy."""
        self._extract_months()
        self._add_holidays()
        self._add_weekdays()
        self._add_seasonal_labels()
        self._add_inverse_relationship()
        self._add_fourier_features()

        Logger.log_info("The data was successfully labelled.")

        new_data_path = root / 'data' / 'backups' / 'labelled_data.csv'
        self._df.to_csv(new_data_path, index=False)
        Logger.log_info(f"The labelled data was saved in {new_data_path}")

        return self._df


class FeatureTransformation:
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Class for transforming features in all numerical values.
        Also handles skewness in the data.

        :param data: the data frame that needs to be labelled and
                     transformed.
        """
        self.labelling = LabellingStrategy(data)
        self._data = self.labelling._run()
        Logger.log_info("Initializing feature transformation...")

    def transform_strings(self) -> pd.DataFrame:
        """
        Transforms features, such as month and weekday names
        into numerical data.
        Transforms boolean features into numerical.
        """
        try:
            month_dict = {
                "January": 1, "February": 2, "March": 3, "April": 4,
                "May": 5, "June": 6, "July": 7, "August": 8, "September": 9,
                "October": 10, "November": 11, "December": 12
            }
            week_dict = {
                "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
                "Friday": 5, "Saturday": 6, "Sunday": 7
            }
            self._data['Month'] = self._data['Month'].map(month_dict)
            self._data['Weekday'] = self._data['Weekday'].map(week_dict)
            Logger.log_info("Successfully transformed" +
                            "month and weekday names.")
        except Exception as e:
            Logger.log_error("Error while transforming month" +
                             f" and weekday names: {str(e)}")
            return pd.DataFrame()

    def transform_boolean(self) -> None:
        """Transform boolean features into numerical."""
        try:
            logical_cols = self._data.select_dtypes(include=['bool']).columns
            self._data[logical_cols] = self._data[logical_cols].astype(int)
            Logger.log_info("Successfully transformed boolean features.")
        except Exception as e:
            Logger.log_error("Error while transforming" +
                             f" boolean features: {str(e)}")
            return pd.DataFrame()

    def detect_skewness(self, threshold: float = 0.75) -> list:
        """
        Identify skewed features that exceed threshold.
        :param threshold: threshold for determining skewness.
        """
        numeric_cols = self._data.select_dtypes(include=[np.number])
        skewed_cols = numeric_cols.skew().index[
            numeric_cols.skew().abs() > threshold].tolist()
        Logger.log_info(f"Detected skewed features: {skewed_cols}")
        return skewed_cols

    def apply_log_transform(self) -> None:
        """Apply log transformation for skewed distributions."""
        try:
            skewed_cols = self.detect_skewness()
            for col in skewed_cols:
                self._data[col] = np.log1p(self._data[col])
            Logger.log_info("Successfully applied the log transformation " +
                            "to the skewed features.")
        except Exception as e:
            Logger.log_error("Error while applying log transformation " +
                             f"to the features: {str(e)}")


class DataNormalizer:
    """Normalizes values within each split to prevent data leakage."""
    def __init__(self) -> None:
        Logger.log_info("Preparing data normalizer.")

    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the dataset with MinMaxScaler.
        :param data: data frame for normalization process.
        """
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            scaler = MinMaxScaler(feature_range=(0, 1))

            # Scale numeric columns only
            scaled_data = scaler.fit_transform(data[numeric_cols])
            scaled_data = pd.DataFrame(scaled_data, columns=numeric_cols,
                                       index=data.index)

            # Keep non-numeric columns (e.g., datetime) unchanged
            non_numeric_cols = data.select_dtypes(exclude=[np.number])
            Logger.log_info("Non-numeric columns retained: " +
                            f"{non_numeric_cols.columns.tolist()}")
            scaled_data = pd.concat([scaled_data, non_numeric_cols], axis=1)

            return scaled_data
        except Exception as e:
            Logger.log_error(f"Error while normalizing the data: {str(e)}")
            return pd.DataFrame()

    def scale_data(self) -> pd.DataFrame:
        """Standardize the data using StandardScaler."""
        try:
            numeric_cols = self._data.select_dtypes(include=[np.number])
            scaler = StandardScaler()
            scaled_df = pd.DataFrame(scaler.fit_transform(numeric_cols),
                                     columns=numeric_cols.columns)

            Logger.log_info("Data successfully scaled using StandardScaler.")

            # add the non-numeric columns back
            non_numeric_cols = self._data.select_dtypes(exclude=[np.number])
            Logger.log_info("Non-numeric columns detected: " +
                            f"{non_numeric_cols.columns.tolist()}")
            scaled_df = pd.concat([scaled_df, non_numeric_cols], axis=1)
            scaled_df = self.reorder_data(scaled_df)

            return scaled_df
        except Exception as e:
            Logger.log_error(f"Error while scaling data: {str(e)}")
            return pd.DataFrame()

    def reorder_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Order the feature columns in the data.
        :param df: data frame used for reordering the features.
        """
        ordered_columns = [
            'Avg_NO2', 'Avg_O3', 'Avg_NO', 'Avg_NOX', 'temp', 'humidity',
            'windspeed', 'cloudcover', 'visibility', 'solarradiation',
            'Month', 'Holiday', 'Weekday', 'DayOfYear',
            'High_O3_warm_months', 'Inverse_NO2_O3',
            'sin_1', 'cos_1', 'sin_2', 'cos_2', 'datetime'
        ]
        missing_columns = [
            col for col in ordered_columns if col not in df.columns
        ]
        if missing_columns:
            Logger.log_error("Missing columns in the " +
                             f"DataFrame: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        df = df[ordered_columns]
        Logger.log_info("Successfully reordered the data.")

        return df


class DataSplitting:
    def __init__(self, data: pd.DataFrame, train_size: float,
                 val_size: float) -> None:
        """
        Initialize the DataSplitting class.

        :param data: The input DataFrame containing the time series data.
        :param train_size: The proportion of the data to use for training.
        :param val_size: The proportion of the data to use for validation.
        """
        self._df = data
        self.train_size = train_size
        self.val_size = val_size
        Logger.log_info("Initializing data splitting for time series...")

    def data_splitting(self) -> MultipleDatasets:
        Logger.log_info("Starting to split the data into train, validation," +
                        "and test sets.")

        # Sort data by date
        self._df.sort_values(by='datetime', inplace=True)

        # Calculate indices for sequential splitting
        train_end = int(len(self._df) * self.train_size)
        val_end = train_end + int(len(self._df) * self.val_size)

        # Split data sequentially
        train_df = self._df.iloc[:train_end]
        val_df = self._df.iloc[train_end:val_end]
        test_df = self._df.iloc[val_end:]

        # Validate no leakage
        if not self.validate_no_leakage(train_df, val_df, test_df):
            Logger.log_error("Data leakage detected between splits.")
            raise ValueError("Data leakage detected between splits.")

        # Save the splits
        split_path = root / 'data' / 'splits'
        split_path.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(split_path / 'train_data.csv', index=False)
        val_df.to_csv(split_path / 'val_data.csv', index=False)
        test_df.to_csv(split_path / 'test_data.csv', index=False)

        Logger.log_info(f"The datasets were saved at {split_path}")
        Logger.log_info("Data splitting completed successfully.")

        return train_df, val_df, test_df

    def validate_no_leakage(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                            test_df: pd.DataFrame) -> bool:
        """Validates that there are no overlapping dates between sets."""
        train_dates = pd.to_datetime(train_df['datetime'])
        val_dates = pd.to_datetime(val_df['datetime'])
        test_dates = pd.to_datetime(test_df['datetime'])

        # ensure validation dates are after training dates
        if val_dates.min() <= train_dates.max():
            Logger.log_error("Data leakage detected: Validation " +
                             "set overlaps with training set.")
            return False
        # ensure test dates are after validation dates
        if test_dates.min() <= val_dates.max():
            Logger.log_error("Data leakage detected: Test set " +
                             "overlaps with validation set.")
            return False

        Logger.log_info("Data leakage validation passed: " +
                        "No overlaps between sets.")
        return True


class CorrelationPlotter:
    """Generates a correlation matrix from the normalized dataset."""
    def __init__(self, normalized_data_path: Path) -> None:
        """
        Initializes the class.
        :param normalized_data_path: path to the normalized data.
        """
        self.normalized_data_path = normalized_data_path
        Logger.log_info("Preparing to generate the correlation matrix...")

    def plot(self) -> None:
        """Load normalized data and plot the correlation matrix heatmap."""
        try:
            normalized_data = pd.read_csv(self.normalized_data_path)
            Logger.log_info("Successfully loaded the normalized" +
                            " data for plotting.")
        except Exception as e:
            Logger.log_error("Error while loading the" +
                             f" normalized data: {str(e)}")
            return pd.DataFrame

        # visualize the columns to check for pollutants
        Logger.log_info(f"Columns in the dataset: {normalized_data.columns}")
        print("Columns in the dataset:", normalized_data.columns)

        # check for NaN values and handle them
        numeric_cols = normalized_data.select_dtypes(
            include='number'
        ).fillna(0)

        # compute the correlation matrix
        correlation_matrix = numeric_cols.corr()

        # plot the heatmap and save it in the features/ folder
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix Heatmap')

        plot_dir = root / 'results' / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / 'correlation_matrix_heatmap.png'

        plt.savefig(plot_path)
        Logger.log_info("The heatmap was saved" +
                        " as 'correlation_matrix_heatmap.png'.")
        print("Saved as 'correlation_matrix_heatmap.png'.")


def main():
    """Main funtion that performs feature engineering."""
    def split_and_normalize(input_data: pd.DataFrame,
                            train_size: float, val_size: float,
                            test_size: float) -> MultipleDatasets:
        """
        Calls DataSplitting and DataNormalizer classes.
        :param input_data: the data frame
        :param train_size: the size for training set
        :param val_size: the size for validation set
        :param test_size: the size for testing set
        :return: Three scaled sets.
        """
        # split data
        split = DataSplitting(input_data, train_size=train_size,
                              val_size=val_size)
        train_data, val_data, test_data = split.data_splitting()

        # normalize each split separately
        normalizer = DataNormalizer()
        scaled_train_data = normalizer.normalize(train_data)
        scaled_val_data = normalizer.normalize(val_data)
        scaled_test_data = normalizer.normalize(test_data)

        return scaled_train_data, scaled_val_data, scaled_test_data

    file_to_preprocess = root / 'data' / 'processed' / 'merged_data.csv'

    loader = DataLoader(file_to_preprocess)
    data = loader.load_data()

    transformer = FeatureTransformation(data)
    transformer.transform_strings()
    transformer.transform_boolean()
    transformer.apply_log_transform()

    # define split ratios
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    # split and normalize data
    scaled_train_data, scaled_val_data, scaled_test_data = \
        split_and_normalize(data, train_size, val_size, test_size)

    train_file_to_save = root / 'data' / 'processed' /\
        'normalized_train_data.csv'
    val_file_to_save = root / 'data' / 'processed' / 'normalized_val_data.csv'
    test_file_to_save = root / 'data' / 'processed' /\
        'normalized_test_data.csv'

    scaled_train_data.to_csv(train_file_to_save, index=False)
    Logger.log_info("The normalized training data " +
                    f"has been saved at {train_file_to_save}")

    scaled_val_data.to_csv(val_file_to_save, index=False)
    Logger.log_info("The normalized validation data " +
                    f"has been saved at {val_file_to_save}")

    scaled_test_data.to_csv(test_file_to_save, index=False)
    Logger.log_info("The normalized test data " +
                    f"has been saved at {test_file_to_save}")

    plotter = CorrelationPlotter(train_file_to_save)
    plotter.plot()
