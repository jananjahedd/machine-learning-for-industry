"""
File: data_pipeline.py
Authors: Janan Jahed (s5107318), Andrei Medesan (s5130727),
         Andjela Matic (s5248736), Stefan Stoian (s4775309)
Description: The file executes the preprocessing.py file.
"""
from pathlib import Path
from preprocessing import DataExtractor, DataPreprocessing
from preprocessing import PreprocessingPipeline


root = Path(__file__).resolve().parent.parent.parent


def main():
    station_names = ['NL00636', 'NL00639', 'NL00643']

    # define the paths
    raw_dir = root / 'data' / 'raw'
    backup_dir = root / 'data' / 'backups'
    processed_dir = root / 'data' / 'processed'

    # check whether the paths exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # set the file name
    raw_file_path = raw_dir / 'eea_data.csv'
    last_date_path = raw_dir / 'last_date.txt'

    extractor = DataExtractor('NL', raw_file_path, last_date_path)
    preprocessor = DataPreprocessing(station_names)
    pipeline = PreprocessingPipeline(extractor, preprocessor,
                                     backup_dir, processed_dir)

    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
