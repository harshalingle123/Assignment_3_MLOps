import numpy as np
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Configuration constants
OUTPUT_DIR = "trained_models"
MODEL_FILENAME = "regression_model.joblib"
TEST_DATA_FILENAME = "holdout_data.joblib"
RANDOM_SEED = 123
SPLIT_RATIO = 0.25

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

def fetch_housing_data():
    """Retrieve and prepare the California Housing dataset."""
    try:
        logging.info("Fetching California Housing dataset...")
        data = fetch_california_housing()
        features, target = data.data, data.target
        logging.info(f"Features shape: {features.shape}, Target shape: {target.shape}")
        logging.info(f"Feature names: {list(data.feature_names)}")
        return features, target
    except Exception as e:
        logging.error(f"Error fetching dataset: {str(e)}")
        raise

def build_and_evaluate_model(features, target):
    """Train and evaluate a Linear Regression model."""
    try:
        logging.info("Preparing train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=SPLIT_RATIO, random_state=RANDOM_SEED
        )

        logging.info("Building Linear Regression model...")
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        train_predictions = regressor.predict(X_train)
        test_predictions = regressor.predict(X_test)

        train_r2_score = r2_score(y_train, train_predictions)
        test_r2_score = r2_score(y_test, test_predictions)
        train_rmse_score = np.sqrt(mean_squared_error(y_train, train_predictions))
        test_rmse_score = np.sqrt(mean_squared_error(y_test, test_predictions))

        logging.info(f"Train R²: {train_r2_score:.4f}, Train RMSE: {train_rmse_score:.4f}")
        logging.info(f"Test R²: {test_r2_score:.4f}, Test RMSE: {test_rmse_score:.4f}")

        return regressor, (X_test, y_test)
    except Exception as e:
        logging.error(f"Model training error: {str(e)}")
        raise

def store_model_and_data(model, test_data):
    """Save the trained model and test data to disk."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_file = os.path.join(OUTPUT_DIR, f"{MODEL_FILENAME}")
        test_data_file = os.path.join(OUTPUT_DIR, f"{TEST_DATA_FILENAME}")

        joblib.dump(model, model_file)
        joblib.dump(test_data, test_data_file)

        logging.info(f"Saved model to: {model_file}")
        logging.info(f"Saved test data to: {test_data_file}")

        logging.debug(f"Model coefficients: {model.coef_}")
        logging.debug(f"Model intercept: {model.intercept_}")
    except Exception as e:
        logging.error(f"Error saving model or data: {str(e)}")
        raise

def run_pipeline():
    """Execute the full model training and saving pipeline."""
    logging.info("Initiating housing price prediction pipeline...")
    features, target = fetch_housing_data()
    trained_model, holdout_data = build_and_evaluate_model(features, target)
    store_model_and_data(trained_model, holdout_data)
    logging.info("Pipeline execution completed.")

if __name__ == "__main__":
    run_pipeline()