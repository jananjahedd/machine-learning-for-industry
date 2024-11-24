import json
import csv
import pandas as pd

def load_metrics(model_name, metrics_dir):
    """Load evaluation metrics from a JSON file."""
    try:
        with open(f"{metrics_dir}/{model_name}_metrics.json", 'r') as f:
            metrics = json.load(f)
        # Return the specific Test metrics as requested
        return {
            'test_mse': metrics["Test"]["MSE"],
            'test_r2': metrics["Test"]["R2"],
            'test_rmse': metrics["Test"]["RMSE"]
        }
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        return None

def get_current_pollutant_data(pollutant_dir):
    """Load current pollutant levels from CSV file."""
    try:
        pollutants = pd.read_csv(pollutant_dir/"merged_data.csv")
        final_line = pollutants.tail(1)
        return final_line
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        return None


def get_predictions(predictions_dir, model_name):
    """Load predicted pollutant levels from CSV file."""
    try:
        with open(f'{predictions_dir}/{model_name}_predictions.csv', newline='') as f:
            reader = csv.reader(f)
            predictions = [list(map(float, row)) for row in list(reader)[:3]]
        print(predictions)
        return predictions
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        return None


def get_health_recommendations(current_data):
    recommendations = []

    # NO2 recommendations
    if current_data['Avg_NO2'].item() > 20:
        recommendations.append("NO2 levels are above the guideline. People with respiratory conditions should take precautions.")

    # O3 recommendations
    if current_data['Avg_O3'].item() > 60:
        recommendations.append("Ozone levels are high. Reduce outdoor activities, especially during midday hours.")

    if not recommendations:
        recommendations.append("Pollutant levels are within safe limits. Enjoy your outdoor activities.")

    return recommendations
