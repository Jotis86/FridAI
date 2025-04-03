"""
Configuration settings for the FridAI application.
This file contains various settings and constants used throughout the application.
"""

# Application settings
APP_NAME = "FridAI"
APP_DESCRIPTION = "No-Code Predictive Modeling Tool"
VERSION = "1.0.0"
AUTHOR = "Jotis"

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