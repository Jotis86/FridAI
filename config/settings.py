"""
Configuration settings for the FridAI application.
This file contains various settings and constants used throughout the application.
"""

import os
from pathlib import Path

# Application settings
APP_NAME = "FridAI"
APP_DESCRIPTION = "No-Code Predictive Modeling Tool"
VERSION = "1.0.0"
AUTHOR = "Jotis"

# File paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMP_DIR = os.path.join(BASE_DIR, "temp")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Model settings
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_FOLDS = 5

# Classification models and their default parameters
CLASSIFICATION_MODELS = {
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "Logistic Regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    "Support Vector Machine": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
    },
    "Gradient Boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
    },
}

# Regression models and their default parameters
REGRESSION_MODELS = {
    "Linear Regression": {},
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "Gradient Boosting": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
    },
    "Support Vector Machine": {
        "kernel": "rbf",
        "gamma": "scale",
        "C": 1.0,
    },
}

# Visualization settings
PLOT_WIDTH = 10
PLOT_HEIGHT = 6
DEFAULT_CMAP = "viridis"
CORRELATION_CMAP = "coolwarm"

# Maximum file size for upload (in MB)
MAX_UPLOAD_SIZE_MB = 50

# Data preprocessing options
SCALING_METHODS = ["None", "Standard Scaler", "Min-Max Scaler", "Robust Scaler"]
ENCODING_METHODS = ["Label Encoding", "One-Hot Encoding"]
IMPUTATION_METHODS = ["Mean/Mode", "Median/Mode", "Most Frequent", "Constant"]

# UI Theme settings
THEME_PRIMARY_COLOR = "#FF4B4B"
THEME_SECONDARY_COLOR = "#0068C9"
THEME_BACKGROUND_COLOR = "#F5F5F5"