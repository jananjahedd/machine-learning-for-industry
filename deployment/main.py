"""
This is the main entry point for the Dash app. It is used to run the Dash 
app locally.
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', \
                                             'results', 'dashboards')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', \
                                             'src', 'data', 'features')))


from src.data.data_pipeline import main as data_pipeline_main
from src.features.feature_engineering import main as features_main
from src.models.linear_regression import main as train_model_main
from results.dashboards.app import app

def run_pipelines():
    logger.info("Starting the data pipeline")
    data_pipeline_main()
    
    logger.info("Starting the feature engineering pipeline")
    features_main()
    
    logger.info("Training the model")
    train_model_main()

if __name__ == '__main__':
    run_pipelines()
    
    logger.info("Running the Dash app")
    app.run_server(use_reloader=False)