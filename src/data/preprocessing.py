"""
file: preprocessing.py
Authors: Janan Jahed (s5107318), Andrei Medesan (s5130727),
         Andjela Matic (s5248736), Stefan Stoian (s4775309)
Description: The file downloads the Airbase data with the recent
values and the meteorological data. It also preprocesses the data
files, by aggregating pollutants per day, dropping columns, handling
missing values and concatenating into one data file.
"""
import os
import pandas as pd
import logging
from pathlib import Path
import airbase
from datetime import datetime, timedelta
import urllib.request
import csv
import codecs


# set the project root dynamically for efficiency
root = Path(__file__).resolve().parent.parent.parent

# set the log directory for information to be stored
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / 'preprocessing_info.log'

# set up the logging format
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


class DataExtractor():
    def __init__(self, country_code: str,
                 raw_file: str, last_row: str) -> None:
        """
        Class that extracts the necessary data from EEA database.
        :param country_code: the code for Netherlands (NL)
        :param raw_file: the path to the EEA data
        :param last_row: the date time of the last row in the data
        """
        self._country_code = country_code
        self._raw_file = raw_file
        self._last_row = last_row

    def extract(self) -> pd.DataFrame:
        """Extract air quality data from EEA using airbase."""
        Logger.log_info("AirbaseDataExtractor initialized for " +
                        f"country code: {self._country_code}")

        # check whether the data already exists and the last row
        if os.path.exists(self._raw_file) and os.path.exists(self._last_row):
            Logger.log_info("Raw data already exists at " +
                            f"{self._raw_file}. Checking for new data...")

            with open(self._last_row, 'r') as f:
                last_timestep = f.read().strip()

            last_date = datetime.strptime(last_timestep,
                                          '%Y-%m-%d %H:%M:%S %z')
            Logger.log_info(f"Last data point was on: {last_date}")

            # request the new data
            client = airbase.AirbaseClient()
            Logger.log_info("Fetching new data from: " +
                            f"{last_date + timedelta(hours=1)}")
            try:
                r = client.request(country=[self._country_code],
                                   year_from=last_date.year,
                                   year_to=datetime.now().year,
                                   update_date=last_date.strftime(
                                       '%Y-%m-%d %H:%M:%S'))
            except Exception as e:
                Logger.log_error(f"Error while fetching new data: {str(e)}")
                return pd.read_csv(self._raw_file)

            # append the new data
            try:
                temp_file = 'temp_new_data.csv'
                r.download_to_file(temp_file)

                new_data = pd.read_csv(temp_file)
                existing_data = pd.read_csv(self._raw_file)

                new_data['DatetimeEnd'] = pd.to_datetime(
                    new_data['DatetimeEnd'])
                existing_data['DatetimeEnd'] = pd.to_datetime(
                    existing_data['DatetimeEnd'])

                filtered_new_data = new_data[
                    new_data['DatetimeEnd'] > last_date
                ]

                if not filtered_new_data.empty:
                    updated_data = pd.concat(
                        [filtered_new_data, existing_data], ignore_index=True
                    )

                    # sort the data as initially
                    updated_data = updated_data.sort_values(by='DatetimeEnd',
                                                            ascending=False)

                    # save the updated data
                    updated_data.to_csv(self._raw_file, index=False)

                    # save the filtered data for dashboard
                    new_data_path = root / 'data' / 'raw' / 'new_eea_data.csv'
                    filtered_new_data.to_csv(new_data_path, index=False)

                    # flag the last date in the data
                    last_end_datetime = updated_data['DatetimeEnd'].max()
                    with open(self._last_row, 'w') as f:
                        f.write(
                            last_end_datetime.strftime('%Y-%m-%d %H:%M:%S %z')
                        )

                    Logger.log_info("Data successfully updated.")

                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        Logger.log_info(f"Temporary file {temp_file} " +
                                        "successfully deleted.")
                    else:
                        Logger.log_warning(f"Temporary file {temp_file} " +
                                           "does not exist.")
                    return updated_data
                else:
                    Logger.log_info("No new data to append.")
                    return pd.read_csv(self._raw_file)

            except Exception as e:
                Logger.log_error(f"Error while appending new data: {str(e)}")
                return pd.read_csv(self._raw_file)

        else:
            Logger.log_info("Fetching initial data from EEA Airbase API.")
            client = airbase.AirbaseClient()

            # request the EEA data for the selected country code
            try:
                r = client.request(country=[self._country_code],
                                   year_from=2022, year_to=2024)
            except Exception as e:
                Logger.log_error(f"Error while fetching data: {str(e)}")
                return pd.DataFrame()

            # download the data into the designated file
            try:
                r.download_to_file(self._raw_file)
                Logger.log_info("Data successfully downloaded" +
                                f" to file: {self._raw_file}")
                data = pd.read_csv(self._raw_file)

                # flag the last row in the data
                last_end_datetime = data['DatetimeEnd'].max()
                with open(self._last_row, 'w') as f:
                    f.write(last_end_datetime)

                Logger.log_info("Initial data successfully downloaded.")
                return data
            except Exception as e:
                Logger.log_error(f"Error while saving data: {str(e)}")
                return pd.DataFrame()

    def get_last_available_date(self) -> str:
        meteo_file = root / 'data' / 'raw' / 'utrecht_meteo.csv'
        data = pd.read_csv(meteo_file)

        # get the maximum date from the existing data
        last_date = pd.to_datetime(data['datetime'], errors='coerce').max()

        # convert it to the standard format
        return last_date.strftime('%Y-%m-%d')

    def extract_meteo_data(self) -> None:
        Logger.log_info("Starting to download the new meteorological data.")

        last_available_date = self.get_last_available_date()
        today_date = datetime.now().strftime('%Y-%m-%d')

        # define the URL and the API key (used mede's key)
        api_key = 'LJ2T3B8CTD25MZCVKX3PPLPW3'
        api_url = ("https://weather.visualcrossing.com/" +
                   "VisualCrossingWebServices/rest/services/timeline/" +
                   f"utrecht/{last_available_date}/{today_date}" +
                   f"?unitGroup=us&include=days&key={api_key}" +
                   "&contentType=csv")

        file_path = root / 'data' / 'raw' / 'new_meteo_data.csv'

        try:
            # download the data
            Logger.log_info("Downloading meteorological data from " +
                            "Visual Crossing API...")
            response = urllib.request.urlopen(api_url)
            CSVText = csv.reader(codecs.iterdecode(response, 'utf-8'))
            Logger.log_info("Meteorological data downloaded successfully.")

            # convert the csv data into a DataFrame
            data = pd.DataFrame(CSVText)

            # ensure column names are assigned properly
            data.columns = data.iloc[0]
            data = data[1:]

            # rename the name column
            data['name'] = data['name'].replace('"Utrecht, Nederland"',
                                                'utrecht')

            Logger.log_info("CSV data converted to DataFrame successfully.")

            data.to_csv(file_path, index=False)
            Logger.log_info(f"Meteorological data saved to {file_path}")

            return data

        except urllib.error.HTTPError as e:
            ErrorInfo = e.read().decode()
            Logger.log_error("HTTPError during meteorological data " +
                             f"download: {ErrorInfo}")
            return pd.DataFrame()
        except urllib.error.URLError as e:
            ErrorInfo = e.reason
            Logger.log_error("URLError during meteorological data " +
                             f"download: {ErrorInfo}")
            return pd.DataFrame()
        except Exception as e:
            Logger.log_error(f"Unexpected error: {str(e)}")
            return pd.DataFrame()


class DataPreprocessing:
    def __init__(self, station_names: list) -> None:
        """
        Preprocesses the EEA data and the Meteorological datasets.
        :param station_names: the station codes for Utrecht.
        """
        self.station_names = station_names

    def preprocess_eea(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the raw data by stations and drop unused columns.
        :param data: eea data for preprocessing
        :return: preprocessed eea dataset
        """
        Logger.log_info("Starting preprocessing of the data.")

        try:
            # filter the data only with the designated stations
            filtered_data = data[
                data['AirQualityStationEoICode'].isin(self.station_names)
            ]
            Logger.log_info("Filtered data with " +
                            f"stations: {self.station_names}")
        except Exception as e:
            Logger.log_error(f"Error while filtering the stations: {str(e)}")
            return pd.DataFrame()

        try:
            # drop unnecessary columns
            columns_drop = ['Countrycode', 'Namespace', 'AirQualityNetwork',
                            'AirQualityStation', 'SamplingPoint',
                            'SamplingProcess', 'Sample', 'AirPollutantCode',
                            'UnitOfMeasurement', 'Validity', 'Verification']
            filtered_data = filtered_data.drop(columns=columns_drop,
                                               errors='ignore')
            Logger.log_info(f"Dropped unnecessary columns: {columns_drop}")
        except Exception as e:
            Logger.log_error(f"Error while dropping columns: {str(e)}")
            return pd.DataFrame()

        try:
            # handle missing values
            if 'Concentration' in filtered_data.columns:
                filtered_data['Concentration'].fillna(
                    filtered_data['Concentration'].mean(), inplace=True
                )
                Logger.log_info("Handled missing values in the EEA dataset.")
            else:
                Logger.log_warning("The 'Concentration' column was " +
                                   "not found in the dataset.")
        except Exception as e:
            Logger.log_error(f"Error while handling missing values: {str(e)}")
            return pd.DataFrame()

        return filtered_data

    def preprocess_meteo(self, file_path: str) -> pd.DataFrame:
        """
        Preprocess the meteorological dataset.
        :param file_path: path to the meteo data.
        :returns: preprocessed meteo data.
        """
        Logger.log_info(f"Preprocessing {file_path} by dropping " +
                        "unnecessary columns...")

        try:
            data = pd.read_csv(file_path)
            Logger.log_info(f"Successfully loaded the data from {file_path}")
            # drop unnecessary columns
            columns_drop = ['name', 'severerisk', 'sunrise',
                            'sunset', 'windgust', 'precipcover',
                            'uvindex', 'solarenergy', 'dew', 'preciptype',
                            'winddir', 'precipprob', 'tempmin',
                            'tempmax', 'sealevelpressure', 'conditions',
                            'description', 'icon', 'stations',
                            'feelslikemax', 'snow', 'feelslike',
                            'feelslikemin', 'snowdepth', 'moonphase']

            preprocessed_data = data.drop(columns=columns_drop,
                                          errors='ignore')
            Logger.log_info(f"Dropped unnecessary columns:{columns_drop}")

            return preprocessed_data

        except Exception as e:
            Logger.log_error("Error while dropping columns " +
                             f"in {file_path}: {str(e)}")
            return pd.DataFrame()


class DataAggregator:
    def __init__(self, processed_data: pd.DataFrame) -> None:
        """
        Aggregates the data per day for EEA dataset.
        :param processed_data: the processed EEA data
        """
        self.processed = processed_data
        Logger.log_info("Initializing Data Aggregator for EEA data...")

    def aggregate(self) -> pd.DataFrame:
        Logger.log_info("Convert the date values for aggregation.")
        self.processed['DatetimeBegin'] = pd.to_datetime(
            self.processed['DatetimeBegin'], errors='coerce'
        )
        Logger.log_info("Aggregating the pollutants by day...")
        self.aggregated_data = self.processed.groupby(
            self.processed['DatetimeBegin'].dt.date).agg(
            Avg_NO=(
                'Concentration',
                lambda x: x[self.processed['AirPollutant'] == 'NO'].mean()
            ),
            Avg_NO2=(
                'Concentration',
                lambda x: x[self.processed['AirPollutant'] == 'NO2'].mean()
            ),
            Avg_O3=(
                'Concentration',
                lambda x: x[self.processed['AirPollutant'] == 'O3'].mean()
            ),
            Avg_NOX=(
                'Concentration',
                lambda x: x[
                    self.processed['AirPollutant'] == 'NOX as NO2'].mean()
                if not x[
                    self.processed['AirPollutant'] == 'NOX as NO2'
                    ].isna().all() else 0
            )
        ).reset_index().rename(columns={'DatetimeBegin': 'Date'})

        Logger.log_info("The pollutants were averaged successfully.")

        return self.aggregated_data


class DataMerger:
    def __init__(self, utrecht_meteo: pd.DataFrame) -> None:
        """
        General class for merging datasets.
        :param utrecht_meteo: the meteorological data
        """
        self.utrecht_meteo = utrecht_meteo
        Logger.log_info("Initializing Data Merger...")

    def remove_duplicates(self, df: pd.DataFrame,
                          column_name: str) -> pd.DataFrame:
        duplicate_count = df.duplicated(subset=column_name).sum()
        if duplicate_count > 0:
            Logger.log_warning(f"Found {duplicate_count} duplicates " +
                               f"in {column_name}. Removing duplicates...")
            df = df.drop_duplicates(subset=column_name)
        return df

    def merge_eea_meteo(self, aggregated_eea: pd.DataFrame) -> pd.DataFrame:
        """
        Merge aggregated EEA data with Utrecht meteorological data.
        :param aggregated_eea: the averaged EEA data
        :return: the merged finalized data
        """
        Logger.log_info("Aligning and concatenating aggregated EEA " +
                        "data with Utrecht meteorological data...")

        if 'Date' not in aggregated_eea.columns:
            Logger.log_info("'Date' column missing in aggregated_eea. " +
                            "Deriving from 'DatetimeBegin'.")
            aggregated_eea['Date'] = pd.to_datetime(
                aggregated_eea['DatetimeBegin'], errors='coerce').dt.date

            if aggregated_eea['Date'].isna().sum() > 0:
                Logger.log_error("Aggregated EEA data contains NaT " +
                                 "values after conversion: "
                                 f"{aggregated_eea['Date'].isna().sum()} rows")
                raise ValueError("Conversion of 'DatetimeBegin' to 'Date' " +
                                 "resulted in NaT values.")

        if 'Date' not in self.utrecht_meteo.columns:
            Logger.log_info("'Date' column missing in utrecht_meteo. " +
                            "Deriving from 'datetime'.")
            self.utrecht_meteo['Date'] = pd.to_datetime(
                self.utrecht_meteo['datetime'], errors='coerce').dt.date

            if self.utrecht_meteo['Date'].isna().sum() > 0:
                Logger.log_error("Aggregated EEA data contains NaT " +
                                 "values after conversion: "
                                 f"{aggregated_eea['Date'].isna().sum()} rows")
                raise ValueError("Conversion of 'datetime' to 'Date' " +
                                 "resulted in NaT values.")

        # remove duplicates in the 'Date' column
        aggregated_eea = self.remove_duplicates(aggregated_eea, 'Date')
        self.utrecht_meteo = self.remove_duplicates(self.utrecht_meteo, 'Date')

        Logger.log_info(f"Aggregated EEA data shape: {aggregated_eea.shape}")
        Logger.log_info("Utrecht meteorological data shape: " +
                        f"{self.utrecht_meteo.shape}")

        # set 'Date' as index for both DataFrames
        aggregated_eea.set_index('Date', inplace=True)
        self.utrecht_meteo.set_index('Date', inplace=True)

        if not aggregated_eea.index.isin(self.utrecht_meteo.index).all():
            Logger.log_warning("Some dates in aggregated EEA data do not " +
                               "have corresponding dates in meteo data.")

        # concatenate the two DataFrames
        concatenated_data = pd.concat(
            [aggregated_eea, self.utrecht_meteo], axis=1, join='outer'
        )

        # fill missing pollutant data
        concatenated_data.ffill(inplace=True)

        # check for missing values after merging
        missing_after_merge = concatenated_data.isna().sum()
        Logger.log_info("Missing values after merging: " +
                        f"{missing_after_merge.to_dict()}")

        concatenated_data = concatenated_data.sort_values(by='Date',
                                                          ascending=False)
        Logger.log_info("The concatenated data was sorted successfully.")

        Logger.log_info("Aggregated EEA data and Utrecht meteorological " +
                        "data concatenated successfully.")
        return concatenated_data

    def merge_new_meteo(self, new_meteo_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge new meteorological data with Utrecht meteorological data.
        :param new_meteo_data: the new data to be merged
        :return: the merged meteo data.
        """
        Logger.log_info("Aligning and concatenating new meteorological " +
                        "data with existing Utrecht meteorological data...")

        if 'Date' not in new_meteo_data.columns:
            Logger.log_info("'Date' column missing in new_meteo_data. " +
                            "Deriving from 'datetime'.")
            new_meteo_data['Date'] = pd.to_datetime(
                new_meteo_data['datetime'], errors='coerce').dt.date

        if 'Date' not in self.utrecht_meteo.columns:
            Logger.log_info("'Date' column missing in utrecht_meteo. " +
                            "Deriving from 'datetime'.")
            self.utrecht_meteo['Date'] = pd.to_datetime(
                self.utrecht_meteo['datetime'], errors='coerce').dt.date

        # remove duplicates in the 'Date' column
        new_meteo_data = self.remove_duplicates(new_meteo_data, 'Date')
        self.utrecht_meteo = self.remove_duplicates(self.utrecht_meteo, 'Date')

        # set 'Date' as index for both DataFrames
        new_meteo_data.set_index('Date', inplace=True)
        self.utrecht_meteo.set_index('Date', inplace=True)

        # concatenate the two DataFrames
        concatenated_data = pd.concat(
            [self.utrecht_meteo, new_meteo_data], axis=0, join='outer'
        )

        # fill missing data
        concatenated_data.ffill(inplace=True)

        Logger.log_info("New meteorological data and Utrecht meteorological " +
                        "data concatenated successfully.")
        return concatenated_data


class PreprocessingPipeline:
    def __init__(self, extractor: DataExtractor,
                 preprocessor: DataPreprocessing,
                 backup_dir: Path, processed_dir: Path) -> None:
        """
        Pipeline class to extract and preprocess data.
        :param extractor: initializes teh DataExtrator class
        :param preprocessor: initializes the DataPreprocessing class
        :param backup_dir: the backup directory path
        :param processed_dir: the processed directory path
        """
        self._extractor = extractor
        self._preprocessor = preprocessor
        self.backup_dir = backup_dir
        self.processed_dir = processed_dir
        self.counter = 1

        # check whether the directories exist otherwise create them
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _create_backup(self, df: pd.DataFrame, stage: str) -> None:
        """
        Create a copy of the data at the given stage.
        :param df: the data frame to save for backup
        :param stage: the stage of the pipeline
        """
        backup_file = self.backup_dir / f'{stage}_backup.csv'
        df.to_csv(backup_file, index=False)
        Logger.log_info(f"Backup created for {stage} stage at {backup_file}")

    def _save_data(self, df: pd.DataFrame, name: str) -> None:
        """
        Save the processed file in the designated directory.
        :param df: the data frame to save
        :param name: the name for the file saved
        """
        processed_file = self.processed_dir / f'{name}.csv'
        df.to_csv(processed_file, index=False)
        Logger.log_info(f"Processed file saved at {processed_file}")

    def run_pipeline(self) -> None:
        """Run the extraction and preprocessing pipeline."""
        Logger.log_info("Running the data extraction and " +
                        "preprocessing pipeline.")

        # extract and preprocess air quality (EEA) data
        Logger.log_info("Step 1: Extracting air quality (EEA) data...")
        raw_eea_data = self._extractor.extract()
        if raw_eea_data.empty:
            Logger.log_error("Air quality data extraction failed. " +
                             "Exiting pipeline...")
            return pd.DataFrame()

        Logger.log_info("Preprocessing air quality (EEA) data...")
        processed_eea_data = self._preprocessor.preprocess_eea(raw_eea_data)
        if processed_eea_data.empty:
            Logger.log_error("Preprocessing EEA data failed. " +
                             "Exiting pipeline...")
            return pd.DataFrame()
        self._save_data(processed_eea_data, 'processed_eea_data')

        # aggregate EEA data
        Logger.log_info("Step 2: Aggregating air quality data...")
        aggregator = DataAggregator(processed_eea_data)
        aggregated_eea_data = aggregator.aggregate()
        self._create_backup(aggregated_eea_data, 'aggregated_eea')

        # extract and preprocess meteorological data
        Logger.log_info("Step 3: Extracting and preprocessing " +
                        "meteorological data...")
        file_path = root / 'data' / 'raw' / 'utrecht_meteo.csv'
        processed_meteo_data = self._preprocessor.preprocess_meteo(file_path)
        if processed_meteo_data.empty:
            Logger.log_error("Preprocessing meteorological data failed. " +
                             "Exiting pipeline...")
            return pd.DataFrame()

        # fetch and merge new meteorological data
        Logger.log_info("Step 4: Fetching and merging " +
                        "new meteorological data...")
        new_meteo_data = self._extractor.extract_meteo_data()
        if new_meteo_data.empty:
            Logger.log_warning("New meteorological data extraction failed. " +
                               "Proceeding with existing data.")
            processed_new = pd.DataFrame()
        else:
            new_path = root / 'data' / 'raw' / 'new_meteo_data.csv'
            processed_new = self._preprocessor.preprocess_meteo(new_path)

        Logger.log_info("Merging the newest downloaded data into the " +
                        "processed meteorological data.")
        merger = DataMerger(processed_meteo_data)
        updated_meteo_data = merger.merge_new_meteo(processed_new)
        self._save_data(updated_meteo_data, 'processed_utrecht_meteo')

        # merge meteorological data with aggregated EEA data
        Logger.log_info("Step 5: Merging EEA data with meteorological data...")
        merger = DataMerger(updated_meteo_data)
        merged_data = merger.merge_eea_meteo(aggregated_eea_data)
        self._save_data(merged_data, 'merged_data')

        Logger.log_info("Data pipeline completed successfully.")
