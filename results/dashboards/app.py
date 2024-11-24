import dash
import logging
from pathlib import Path
from layouts import create_layout
from utils import load_metrics, get_predictions, get_current_pollutant_data,\
      get_health_recommendations
import shutil

# Initialize the logger
logging.basicConfig(
    filename='logs/dashboard.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

root = Path(__file__).resolve().parent.parent.parent
log_dir = root / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir / 'dashboard.log'

# Initialize the app
app = dash.Dash(__name__, assets_folder='assets')
logger.info("Dash app initialized")

# Load metrics and dreictories
metrics_dir = root / 'results'
linear_metrics = load_metrics('linear_regression', metrics_dir)
pollutant_dir = root / 'data' / 'processed'
predictions_dir = root / 'results' / 'predictions'

# create assets directory in the root
assets_dir = root / 'results' / 'dashboards' / 'assets'
assets_dir.mkdir(exist_ok=True)
logger.info(f"Assets directory created or already exists: {assets_dir}")

# copy the images from results/plots to assets
for image in ['linear_regression_plot.png', 'random_forest_plot.png',
              'correlation_matrix_heatmap.png']:
    image_path = metrics_dir / 'plots' / image
    dest_image_path = assets_dir / image
    if image_path.exists():
        shutil.copy(image_path, dest_image_path)
        logger.info(f"Copied {image_path} to {dest_image_path}")
    else:
        logger.error(f"Image {image_path} does not exist. " +
                     "Skipping this image.")

current_data = get_current_pollutant_data(pollutant_dir)
predictions = get_predictions(predictions_dir, 'linear_regression')
health_recommendations = get_health_recommendations(current_data)

# Create the layout
app.layout = create_layout(linear_metrics, current_data, predictions,
                           assets_dir, health_recommendations)

if __name__ == '__main__':
    logger.info("Starting the Dash server...")
    app.run_server(debug=True)
