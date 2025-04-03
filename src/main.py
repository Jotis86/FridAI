import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
import pickle
import io
import base64
import sys
import os

# Add parent directory to path to find the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Import settings
    from config.settings import (
        APP_NAME, APP_DESCRIPTION, VERSION, AUTHOR,
        DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE,
        CLASSIFICATION_MODELS, REGRESSION_MODELS,
        PLOT_WIDTH, PLOT_HEIGHT, CORRELATION_CMAP
    )
except ImportError:
    # Default values if settings import fails
    APP_NAME = "FridAI"
    APP_DESCRIPTION = "No-Code Predictive Modeling Tool"
    VERSION = "1.0.0"
    AUTHOR = "Jotis"
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_TEST_SIZE = 0.2
    CLASSIFICATION_MODELS = {
        "Random Forest": {"n_estimators": 100, "max_depth": None},
        "Logistic Regression": {"C": 1.0, "max_iter": 1000},
        "Support Vector Machine": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
        "Gradient Boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    }
    REGRESSION_MODELS = {
        "Linear Regression": {},
        "Random Forest": {"n_estimators": 100, "max_depth": None},
        "Support Vector Machine": {"kernel": "rbf", "gamma": "scale", "C": 1.0},
        "Gradient Boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    }
    PLOT_WIDTH = 10
    PLOT_HEIGHT = 6
    CORRELATION_CMAP = "coolwarm"

# Set page configuration
st.set_page_config(
    page_title=f"{APP_NAME} - {APP_DESCRIPTION}",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to download model as pickle file
def get_model_download_link(model_dict, filename="model.pkl"):
    """Generates a link to download the model"""
    buffer = io.BytesIO()
    pickle.dump(model_dict, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:file/pickle;base64,{b64}" download="{filename}">Download Trained Model (.pkl)</a>'
    return href

# Obtener la ruta absoluta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to images
principal_image_path = os.path.join("images", 'portada.png')
menu_image_path = os.path.join("images", 'frida.png')

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_filename' not in st.session_state:
    st.session_state.model_filename = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

# Configuración de la barra lateral
try:
    st.sidebar.image(menu_image_path, use_container_width=True)
except Exception as e:
    st.sidebar.error(f"Error loading image: {e}")
    st.sidebar.info(f"Looking for image at: {menu_image_path}")

# Sidebar mejorado y personalizado
st.sidebar.title(f"✨ {APP_NAME}")
st.sidebar.markdown(f"<p style='font-size: 18px; font-style: italic; color: #4d8b90;'>{APP_DESCRIPTION}</p>", unsafe_allow_html=True)

# App navigation
page = st.sidebar.radio("", ["Upload Data", "Explore Data", "Train Model", "Download Model"])

# Botón de GitHub estilizado en verde
st.sidebar.markdown("""
<a href='https://github.com/Jotis86/FridAI' target='_blank'>
    <button style='background-color: #2ea44f; border: none; color: white; padding: 10px 24px; 
    text-align: center; text-decoration: none; display: inline-block; font-size: 16px; 
    margin: 4px 2px; cursor: pointer; border-radius: 8px; width: 100%;'>
        <svg style="vertical-align: middle; margin-right: 10px;" height="20" width="20" viewBox="0 0 16 16" fill="white">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
        </svg>
        GitHub Repository
    </button>
</a>
""", unsafe_allow_html=True)

# Sección personalizada de información del creador
st.sidebar.markdown(f"""
<div style='background-color: #f5f7f9; padding: 10px; border-radius: 8px; margin-top: 10px;'>
    <h4 style='color: #333; margin-bottom: 5px;'>Created with 💙</h4>
    <p style='color: #666; margin-bottom: 5px; font-size: 14px;'>by <span style='font-weight: bold; color: #2c3e50;'>{AUTHOR}</span></p>
    <p style='color: #888; font-size: 12px; margin-top: 5px;'>© 2023 {APP_NAME} - All rights reserved</p>
</div>
""", unsafe_allow_html=True)

# Upload Data page
if page == "Upload Data":
    # Mostrar imagen principal
    try:
        st.image(principal_image_path, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.info(f"Looking for image at: {principal_image_path}")

    # Texto explicativo de la aplicación
    st.markdown(f"""
    ## Welcome to {APP_NAME}!

    {APP_NAME} is an interactive tool designed to simplify predictive modeling without writing code.
    Whether you're a data scientist, analyst, or student, this application helps you:

    - **Upload your data** and get immediate insights
    - **Explore and visualize** your data through various charts
    - **Train powerful machine learning models** with a few clicks
    - **Customize parameters** to improve model performance
    - **Download your trained model** for use in other applications
    - **Make predictions** on new data without coding

    Simply upload your CSV file to get started!
    """)
    
    st.title("Upload Your Data")
    st.write("Upload a CSV file to get started with predictive modeling.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.success(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")
            
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            st.subheader("Data Information")
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
            
            st.subheader("Statistical Summary")
            st.write(data.describe())
            
            st.subheader("Missing Values")
            missing = data.isnull().sum()
            st.write(missing[missing > 0])
            
        except Exception as e:
            st.error(f"Error: {e}")

# Explore Data page
elif page == "Explore Data":
    st.title("Explore Your Data")
    
    if st.session_state.data is None:
        st.warning("Please upload data first.")
    else:
        data = st.session_state.data
        
        st.subheader("Data Visualization")
        
        # Select columns for visualization
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        viz_type = st.selectbox("Select Visualization Type", 
                               ["Distribution", "Correlation", "Categorical Analysis"])
        
        if viz_type == "Distribution":
            if numeric_cols:
                col = st.selectbox("Select Numeric Column", numeric_cols)
                
                fig, ax = plt.subplots(1, 2, figsize=(PLOT_WIDTH, PLOT_HEIGHT))
                sns.histplot(data[col], kde=True, ax=ax[0])
                ax[0].set_title(f'Distribution of {col}')
                
                sns.boxplot(y=data[col], ax=ax[1])
                ax[1].set_title(f'Boxplot of {col}')
                
                st.pyplot(fig)
                
                st.write(f"**Statistics for {col}:**")
                st.write(data[col].describe())
            else:
                st.warning("No numeric columns found in the dataset.")
        
        elif viz_type == "Correlation":
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
                corr_matrix = data[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap=CORRELATION_CMAP, ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
        
        elif viz_type == "Categorical Analysis":
            if categorical_cols:
                cat_col = st.selectbox("Select Categorical Column", categorical_cols)
                
                fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
                value_counts = data[cat_col].value_counts()
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Count of {cat_col}')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.write(f"**Value Counts for {cat_col}:**")
                st.write(value_counts)
            else:
                st.warning("No categorical columns found in the dataset.")

# Train Model page
elif page == "Train Model":
    st.title("Train Predictive Model")
    
    if st.session_state.data is None:
        st.warning("Please upload data first.")
    else:
        data = st.session_state.data
        
        st.subheader("Select Target and Features")
        
        # Target variable selection
        target = st.selectbox("Select Target Variable", data.columns)
        st.session_state.target = target
        
        # Problem type detection
        if data[target].dtype == 'object' or data[target].nunique() < 10:
            default_problem = "Classification"
        else:
            default_problem = "Regression"
        
        problem_type = st.radio(
            "Problem Type", 
            ["Classification", "Regression"],
            index=0 if default_problem == "Classification" else 1
        )
        st.session_state.problem_type = problem_type
        
        # Feature selection
        all_features = [col for col in data.columns if col != target]
        features = st.multiselect("Select Features", all_features, default=all_features)
        st.session_state.features = features
        
        # Data preprocessing options
        st.subheader("Data Preprocessing")
        handle_missing = st.checkbox("Handle Missing Values (replace with mean/mode)", value=True)
        
        scaling_option = st.selectbox(
            "Apply Feature Scaling", 
            ["None", "Standard Scaler", "Min-Max Scaler", "Robust Scaler"]
        )
        
        # Encoding option for categorical features
        if data.select_dtypes(include=['object']).columns.tolist():
            encoding_option = st.selectbox(
                "Categorical Encoding Method",
                ["Label Encoding", "One-Hot Encoding"]
            )
        else:
            encoding_option = "Label Encoding"  # Default if no categorical features
        
        test_size = st.slider("Test Set Size (%)", 10, 50, int(DEFAULT_TEST_SIZE*100)) / 100
        
        # Model selection
        st.subheader("Model Selection")
        if problem_type == "Classification":
            model_type = st.selectbox(
                "Select Model", 
                list(CLASSIFICATION_MODELS.keys())
            )
            
            # Show model parameters
            with st.expander("Model Parameters"):
                model_params = {}
                default_params = CLASSIFICATION_MODELS[model_type]
                
                for param, default in default_params.items():
                    if param == "n_estimators":
                        model_params[param] = st.slider(f"{param}", 10, 500, default)
                    elif param == "max_depth":
                        max_depth = st.slider(f"{param}", 1, 50, 10 if default is None else default)
                        model_params[param] = None if max_depth == 50 else max_depth
                    elif param == "min_samples_split":
                        model_params[param] = st.slider(f"{param}", 2, 20, default)
                    elif param == "min_samples_leaf":
                        model_params[param] = st.slider(f"{param}", 1, 20, default)
                    elif param == "C":
                        model_params[param] = st.slider(f"{param}", 0.1, 10.0, float(default))
                    elif param == "max_iter":
                        model_params[param] = st.slider(f"{param}", 100, 2000, default)
                    elif param == "learning_rate":
                        model_params[param] = st.slider(f"{param}", 0.01, 1.0, float(default))
                    else:
                        model_params[param] = default
        else:
            model_type = st.selectbox(
                "Select Model", 
                list(REGRESSION_MODELS.keys())
            )
            
            # Show model parameters
            with st.expander("Model Parameters"):
                model_params = {}
                default_params = REGRESSION_MODELS[model_type]
                
                for param, default in default_params.items():
                    if param == "n_estimators":
                        model_params[param] = st.slider(f"{param}", 10, 500, default)
                    elif param == "max_depth":
                        max_depth = st.slider(f"{param}", 1, 50, 10 if default is None else default)
                        model_params[param] = None if max_depth == 50 else max_depth
                    elif param == "min_samples_split":
                        model_params[param] = st.slider(f"{param}", 2, 20, default)
                    elif param == "min_samples_leaf":
                        model_params[param] = st.slider(f"{param}", 1, 20, default)
                    elif param == "C":
                        model_params[param] = st.slider(f"{param}", 0.1, 10.0, float(default))
                    elif param == "learning_rate":
                        model_params[param] = st.slider(f"{param}", 0.01, 1.0, float(default))
                    else:
                        model_params[param] = default
                        
        # Custom model filename
        model_filename = st.text_input(
            "Model Filename", 
            value=f"{model_type.lower().replace(' ', '_')}_{problem_type.lower()}_model.pkl"
        )
        
        # Train model button
        if st.button("Train Model"):
            if not features:
                st.error("Please select at least one feature.")
            else:
                try:
                    # Prepare data
                    X = data[features].copy()
                    y = data[target].copy()
                    
                    # Handle missing values
                    if handle_missing:
                        for col in X.columns:
                            if X[col].dtype.name in ['float64', 'int64']:
                                X[col].fillna(X[col].mean(), inplace=True)
                            else:
                                X[col].fillna(X[col].mode()[0], inplace=True)
                    
                    # Encode categorical features
                    categorical_features = X.select_dtypes(include=['object']).columns
                    encoder_dict = {}
                    
                    if encoding_option == "Label Encoding":
                        for col in categorical_features:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col])
                            encoder_dict[col] = le
                    else:  # One-Hot Encoding
                        X = pd.get_dummies(X, columns=categorical_features)
                    
                    # Encode target if classification
                    target_encoder = None
                    if problem_type == "Classification" and y.dtype == 'object':
                        target_encoder = LabelEncoder()
                        y = target_encoder.fit_transform(y)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=DEFAULT_RANDOM_STATE
                    )
                    
                    # Apply scaling if selected
                    scaler = None
                    if scaling_option != "None":
                        if scaling_option == "Standard Scaler":
                            scaler = StandardScaler()
                        elif scaling_option == "Min-Max Scaler":
                            scaler = MinMaxScaler()
                        elif scaling_option == "Robust Scaler":
                            scaler = RobustScaler()
                            
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    # Train model based on selection
                    with st.spinner('Training model...'):
                        if problem_type == "Classification":
                            if model_type == "Random Forest":
                                model = RandomForestClassifier(random_state=DEFAULT_RANDOM_STATE, **model_params)
                            elif model_type == "Logistic Regression":
                                model = LogisticRegression(random_state=DEFAULT_RANDOM_STATE, **model_params)
                            elif model_type == "Support Vector Machine":
                                model = SVC(random_state=DEFAULT_RANDOM_STATE, probability=True, **model_params)
                            elif model_type == "Gradient Boosting":
                                model = GradientBoostingClassifier(random_state=DEFAULT_RANDOM_STATE, **model_params)
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            st.success(f"Model trained successfully! Accuracy: {accuracy:.4f}")
                            
                            # Display confusion matrix
                            st.subheader("Confusion Matrix")
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            plt.xlabel('Predicted')
                            plt.ylabel('Actual')
                            plt.title('Confusion Matrix')
                            st.pyplot(fig)
                            
                            # Display classification report
                            st.subheader("Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            st.dataframe(pd.DataFrame(report).transpose())
                            
                        else:  # Regression
                            if model_type == "Random Forest":
                                model = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, **model_params)
                            elif model_type == "Linear Regression":
                                model = LinearRegression(**model_params)
                            elif model_type == "Support Vector Machine":
                                model = SVR(**model_params)
                            elif model_type == "Gradient Boosting":
                                model = GradientBoostingRegressor(random_state=DEFAULT_RANDOM_STATE, **model_params)
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            st.success(f"Model trained successfully! MSE: {mse:.4f}, R²: {r2:.4f}")
                            
                            # Plot actual vs predicted
                            fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
                            ax.scatter(y_test, y_pred, alpha=0.5)
                            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                            ax.set_xlabel('Actual')
                            ax.set_ylabel('Predicted')
                            ax.set_title('Actual vs Predicted Values')
                            st.pyplot(fig)
                    
                    # Save model and metadata in session state
                    model_data = {
                        'model': model,
                        'features': features,
                        'problem_type': problem_type,
                        'model_type': model_type,
                        'scaler': scaler,
                        'encoders': encoder_dict,
                        'target_encoder': target_encoder,
                        'original_columns': list(X.columns),
                        'encoding_method': encoding_option,
                        'scaling_method': scaling_option,
                        'metrics': {
                            'accuracy': accuracy if problem_type == "Classification" else None,
                            'mse': mse if problem_type == "Regression" else None,
                            'r2': r2 if problem_type == "Regression" else None
                        }
                    }
                    
                    st.session_state.model = model_data
                    st.session_state.model_filename = model_filename
                    st.session_state.trained = True
                    
                    # Provide download link
                    st.subheader("Download Model")
                    st.markdown(get_model_download_link(model_data, model_filename), unsafe_allow_html=True)
                    
                    # Feature importance for tree-based models
                    if model_type in ["Random Forest", "Gradient Boosting"]:
                        st.subheader("Feature Importance")
                        # For one-hot encoded data, we need to map back to original feature names
                        if encoding_option == "One-Hot Encoding" and len(categorical_features) > 0:
                            # Get feature names after one-hot encoding
                            feature_names = X.columns.tolist()
                        else:
                            feature_names = features
                            
                        feature_imp = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
                        sns.barplot(x='Importance', y='Feature', data=feature_imp.head(20), ax=ax)
                        plt.title('Feature Importance (Top 20)')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.dataframe(feature_imp)
                
                except Exception as e:
                    st.error(f"Error training model: {e}")
                    st.error(f"Exception details: {str(e)}")

# Download Model page
elif page == "Download Model":
    st.title("Download Trained Model")
    
    if not st.session_state.trained:
        st.warning("Please train a model first before downloading.")
    else:
        st.success("Model is ready for download!")
        
        model_info = st.session_state.model
        model_filename = st.session_state.model_filename
        
        st.subheader("Model Information")
        
        # Display model details
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Model Type:**", model_info['model_type'])
            st.write("**Problem Type:**", model_info['problem_type'])
            st.write("**Features:**", len(model_info['features']))
            
            # Display metrics
            if model_info['problem_type'] == "Classification":
                st.write("**Accuracy:**", f"{model_info['metrics']['accuracy']:.4f}")
            else:
                st.write("**MSE:**", f"{model_info['metrics']['mse']:.4f}")
                st.write("**R²:**", f"{model_info['metrics']['r2']:.4f}")
                
        with col2:
            st.write("**Scaling Method:**", model_info['scaling_method'])
            st.write("**Encoding Method:**", model_info['encoding_method'])
        
        # Download button
        st.subheader("Download Model")
        st.markdown(get_model_download_link(model_info, model_filename), unsafe_allow_html=True)
        
        # Usage instructions
        st.subheader("How to Use the Model")
        st.markdown("""
        ```python
        import pickle
        
        # Load the model
        with open('model_filename.pkl', 'rb') as f:
            model_data = pickle.load(f)
            
        # Extract components
        model = model_data['model']
        scaler = model_data['scaler']
        encoders = model_data['encoders']
        
        # Prepare your data (must have the same features as training data)
        # Apply the same preprocessing steps
        
        # Make predictions
        predictions = model.predict(your_data)
        ```
        """)
        
        # Allow renaming the model file
        new_filename = st.text_input("Rename model file", value=model_filename)
        if new_filename != model_filename and st.button("Update filename"):
            st.session_state.model_filename = new_filename
            st.success(f"Model filename updated to {new_filename}")
            
            # Update download link
            st.markdown(get_model_download_link(model_info, new_filename), unsafe_allow_html=True)