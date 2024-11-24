# Monitoring and Forecasting Air Pollution in the Netherlands
### Authors: Janan Jahed (s5107318), Andrei Medesan (s5130727), Andjela Matic (s5248736), Stefan Stoian (s4775309)
### Group 12

In this project, we design machine learning models that monitor air pollution in the Netherlands, specifically for Utrecht. For this, we developed a pipeline for processing and predicting the nitrogen dioxide (NO2) and ozone (O3) concentrations. The raw datasets are stored in the `raw/` folder of the `data/` directory of the repository. In this directory, the user can also view the processed datasets, and for safety reasons, we implemented copies of the data in the `backups` folder.

In the current state of our pollution predictor project, we have implemented the following files, alongside their functionalities in the designated folders as follows:

The `notebooks/` folder contains the Jupyter notebook of the data distribution used to view plots and get more insight into our data for performing feature engineering. The notebook description can be seen below:

`data_distribution.ipynb`: In this file, the user can observe the datasets' descriptions and also regard histograms and heatmaps in each dataset. This allows the users to understand the data distribution before performing feature engineering. In this jupyter notebook, users can expect to see the following:

    - The datasets' details such as number of entries, basic statistics or the data head.
    - Histograms for NO2 and O3 levels, as well as the heatmap between each pollutant in the EEA dataset.
    - Histograms for features, as well as the heatmap in the Meteorological dataset.
    - Histograms, boxplots and time-series plots of the merged dataset.


In the `src/` directory, the following files were implemented for our forecasting ai pollution project in the respective folders:

1) The `data/` folder contains the preprocessing files that handle the raw datasets and moves the processed data in the designated folder `data/processed/`. The two existing files and their functionality are as follows:

`preprocessing.py`: This script preprocesses the data for the upcoming model. It performs several tasks, including:

    - Downloading the raw `.csv` file from the European Environment Agency (EEA) database.
    - Update the EEA dataset with the most recent recordings for pollutants.
    - Extract the most recent recording for meteorological dataset using a private API key.
    - Preprocess the datasets by dropping unnecessary columns and filter them for Utrecht stations.
    - Aggregating the EEA pollutants per day.
    - Merging the EEA dataset with the Meteorlogical concatenated data into one file called `merged_data.csv`.
    - Logging actions during script execution.

`data_pipeline.py`: This script executes the `preprocessing.py` file for efficient downloading of the EEA data from Airbase.

To run the preprocessing phase, the user must execute the file `data_pipeline.py`.

2) The `features/` folder contains the feature engineering file that processes the data further by transforming features, adding labels for enahncing the data, applying normalization procedure, and splitting the data for feeding the models. The functionalities of the existing file in the folder are as follows:

`feature-engineering.py`: This script performs several tasks, including:

    - Logging actions during script execution.
    - Loading the `merged_data.csv` file from processed/ folder for processing.
    - Adding labels to the data, such as month and weekday names, flags the warmer months for high O3 concentrations, flags the inverse relationship between NO2 and O3 concentrations (when O3 increases, NO2 deacreases), adds sinusoidal pattern of O3 and NO2 trends.
    - Transforming the features into numerical values for efficient training of the models.
    - Checks for skewness in the dataset and solves it by converging the data into normal distribution.
    - Splitting the data into training, testing and validation sets and saving them in `data/splits/` folder.
    - Normalizing each set from above using MinMax Scaler and saves them in the `processed/` folder.
    - Generates the heatmap of the features in merged data and saves it in `results/plots` folder.


3) The `models/` folder contains the three models developed, namely Linear Regression, Random Forest, and Long-Short Term Memory model. All models have logging details stored in `logs/` folder. The functionalities of the three existing files are as follows:

`linear_regression.py`: This script performs linear regression on the data, by training, testing and validating on the designated datasets from `processed/` folder. The user can choose between Lasso or Ridge regularization for training the model. The file produces a predictions CSV file that is saved in `results/predictions/` and the plots for the predictions vs actual values in `results/plots/` directory. The training is performed using K-Fold cross-validation to improve performance and reduce overfitting.

`random_forest.py`: This script builds a Random Forest model (RF) that is trained on the designated data. It has a Grid Search functionality that searches for best hyperparameters that are saved in a JSON file in the same folder as `random_forest_params.json`. This way, the model can be executed multiple times without performing grid search, reducing the computational complexity. It saves the predictions as a CSV file in `results/predictions/` folder and saves the plots corresponding to RF in `results/plots/` directory.  The training is performed using K-Fold cross-validation to improve performance and reduce overfitting.

`lstm.py`: The script builds an LSTM model that is trained on the training dataset. It has the following functionalities:

    - The LSTM model is designed in PyTorch to predict pollutant levels (NO2 and O3) based on historical data.
    - Key parameters include the number of layers, hidden units, and dropout for regularization.
    - Training is performed using K-Fold Cross-Validation, providing robust performance evaluation across folds.
    - Early stopping is applied to prevent overfitting, halting training when validation loss does not improve.
    - Model performance is evaluated on test data using metrics such as MSE, RMSE, and \( R^2 \).
    - Predictions are plotted vs actual values for visual analysis of model accuracy and saved in `results/plots/` directory.
    - Future predictions for NO2 and O3 levels are generated for the next three days based on the latest available data in the terminal.

In the `results/` folder, we have implemented the dashboard files, alongside the app found in `dashboards/` directory, as well as two other folders with the plots and predictions as previously discussed.

The dashboards/ directory has the following files:

`app.py`: This file initializes and runs the main application. It brings together the layout, callbacks, and utility functions to display an interactive dashboard.
   - Runs the Dash app, setting up server configurations and launching the web-based interface for users.
   - After executing the file, the app will start on a local server, accessible in a web browser.

`callbacks.py`: The file contains all callback functions for dynamic interactivity in the dashboard. These callbacks update the visualizations and handle user interactions. It controls interactivity in the dashboard, such as updating graphs and metrics based on user input. This file is automatically imported by `app.py`.

`layouts.py`: The script defines the layout of the dashboard, organizing the layout structure, including buttons, graphs, and visual elements. It sets up the visual components and page structure for the dashboard. Additionally, this file is called by `app.py` to render the visual interface, so no separate action is needed.

`utils.py`: The script includes helper functions for data processing and any calculations required by other files, such as loading data or transforming inputs for the model. It provides reusable utility functions to simplify code across other modules and it is automatically used by `app.py` and other files.


Before you can run the scripts, make sure that:

- You have Python installed. You can download it from python.org. The version should be compatible with the project's dependencies.
- `pip` as is the package installer for Python. Usually comes bundled with Python when you install it.
- Any type of `conda` manager for Python is installed locally.
- Make sure that, before creating any `conda` environment, you are in the root directory of the repository.

From here, the following steps must be made:

1) Create a `conda` environment:
    conda env create -f environment.yml 

2) `conda` now has created an environment. Make sure to check and replace `your_env_name` with the actual name of the environment. Now do:

    conda activate your_env_name

3) Finally, run the following command:

    python deployment/main.py

Be patient, as there is a lot to download and compute! Once the program finishes, you should see the following lines in the terminal:

    * Serving Flask app 'results.dashboards.app'
    * Debug mode: off
    INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
    * Running on http://127.0.0.1:8050

4) Open the link and enjoy! 