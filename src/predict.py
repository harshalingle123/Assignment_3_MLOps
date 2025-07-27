import numpy as np
import joblib
import os
import logging
import sys
from sklearn.metrics import r2_score, mean_squared_error

# Configuration constants
SAVE_DIR = "trained_models"
MODEL_SUFFIX = "regression_model.joblib"
TEST_DATA_SUFFIX = "holdout_data.joblib"
R2_THRESHOLD = 0.55

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def get_latest_files():
    """Retrieve paths to the most recent model and test data files."""
    try:
        logging.info(f"Searching for files in {SAVE_DIR}...")
        model_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(MODEL_SUFFIX)]
        test_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(TEST_DATA_SUFFIX)]
        
        if not model_files or not test_files:
            raise FileNotFoundError(f"No model or test data files found in {SAVE_DIR}")
        
        # Select the latest files by sorting (assumes timestamp in filename)
        latest_model = os.path.join(SAVE_DIR, sorted(model_files)[-1])
        latest_test_data = os.path.join(SAVE_DIR, sorted(test_files)[-1])
        
        logging.info(f"Selected model file: {latest_model}")
        logging.info(f"Selected test data file: {latest_test_data}")
        return latest_model, latest_test_data
    except Exception as e:
        logging.error(f"Failed to locate files: {str(e)}")
        raise

def load_resources():
    """Load the model and test dataset from the latest files."""
    try:
        model_path, test_data_path = get_latest_files()
        
        logging.info("Loading resources...")
        model = joblib.load(model_path)
        test_X, test_y = joblib.load(test_data_path)
        
        logging.info(f"Loaded model and test data successfully.")
        logging.info(f"Test dataset — Features: {test_X.shape}, Target: {test_y.shape}")
        return model, test_X, test_y
    except Exception as e:
        logging.error(f"Resource loading failed: {str(e)}")
        raise

def perform_prediction(model, test_X, test_y):
    """Run predictions and evaluate model performance."""
    try:
        logging.info("Executing predictions...")
        predicted_y = model.predict(test_X)
        
        r2_metric = r2_score(test_y, predicted_y)
        rmse_metric = np.sqrt(mean_squared_error(test_y, predicted_y))
        
        logging.info(f"Evaluation — R²: {r2_metric:.4f}, RMSE: {rmse_metric:.4f}")
        
        logging.info("\nPrediction Samples:")
        logging.info("Actual\t\tPredicted")
        for i in range(min(5, len(test_y))):
            logging.info(f"{test_y[i]:.2f}\t\t{predicted_y[i]:.2f}")
        
        return r2_metric, rmse_metric
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise

def verify_performance():
    """Verify model performance against the R² threshold."""
    logging.info("Starting model performance verification...")
    
    try:
        model, test_X, test_y = load_resources()
        r2, _ = perform_prediction(model, test_X, test_y)
        
        if r2 >= R2_THRESHOLD:
            logging.info(f"Verification SUCCEEDED (R² = {r2:.4f} >= {R2_THRESHOLD})")
            sys.exit(0)
        else:
            logging.error(f"Verification FAILED (R² = {r2:.4f} < {R2_THRESHOLD})")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Verification aborted due to error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    verify_performance()