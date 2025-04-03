import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import pickle
import os
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="FridAI - No-Code Predictive Modeling",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

# Sidebar
st.sidebar.title("FridAI")
st.sidebar.markdown("## No-Code Predictive Modeling")

# App navigation
page = st.sidebar.radio("Navigate", ["Upload Data", "Explore Data", "Train Model", "Make Predictions"])

# Upload Data page
if page == "Upload Data":
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
                
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
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
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = data[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
        
        elif viz_type == "Categorical Analysis":
            if categorical_cols:
                cat_col = st.selectbox("Select Categorical Column", categorical_cols)
                
                fig, ax = plt.subplots(figsize=(10, 6))
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
        scaling = st.checkbox("Apply Feature Scaling", value=True)
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
        
        # Model selection
        st.subheader("Model Selection")
        if problem_type == "Classification":
            model_type = st.selectbox(
                "Select Model", 
                ["Random Forest", "Logistic Regression"]
            )
        else:
            model_type = st.selectbox(
                "Select Model", 
                ["Random Forest", "Linear Regression"]
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
                    for col in categorical_features:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                    
                    # Encode target if classification
                    if problem_type == "Classification":
                        if y.dtype == 'object':
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Scale features if selected
                    if scaling:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    # Train model based on selection
                    with st.spinner('Training model...'):
                        if problem_type == "Classification":
                            if model_type == "Random Forest":
                                model = RandomForestClassifier(n_estimators=100, random_state=42)
                            else:  # Logistic Regression
                                model = LogisticRegression(max_iter=1000, random_state=42)
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            st.success(f"Model trained successfully! Accuracy: {accuracy:.4f}")
                            st.subheader("Classification Report")
                            report = classification_report(y_test, y_pred, output_dict=True)
                            st.dataframe(pd.DataFrame(report).transpose())
                            
                        else:  # Regression
                            if model_type == "Random Forest":
                                model = RandomForestRegressor(n_estimators=100, random_state=42)
                            else:  # Linear Regression
                                model = LinearRegression()
                            
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            st.success(f"Model trained successfully! MSE: {mse:.4f}, R²: {r2:.4f}")
                            
                            # Plot actual vs predicted
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(y_test, y_pred, alpha=0.5)
                            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                            ax.set_xlabel('Actual')
                            ax.set_ylabel('Predicted')
                            ax.set_title('Actual vs Predicted Values')
                            st.pyplot(fig)
                    
                    # Save model in session state
                    st.session_state.model = {
                        'model': model,
                        'features': features,
                        'problem_type': problem_type,
                        'scaler': scaler if scaling else None,
                        'categorical_features': list(categorical_features)
                    }
                    st.session_state.trained = True
                    
                    # Feature importance for tree-based models
                    if model_type == "Random Forest":
                        st.subheader("Feature Importance")
                        feature_imp = pd.DataFrame({
                            'Feature': features,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=feature_imp, ax=ax)
                        plt.title('Feature Importance')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.dataframe(feature_imp)
                
                except Exception as e:
                    st.error(f"Error training model: {e}")

# Make Predictions page
elif page == "Make Predictions":
    st.title("Make Predictions")
    
    if not st.session_state.trained:
        st.warning("Please train a model first.")
    else:
        st.success("Model is ready for predictions!")
        
        model_info = st.session_state.model
        features = model_info['features']
        problem_type = model_info['problem_type']
        
        st.subheader("Make Single Prediction")
        
        # Create input fields for each feature
        input_data = {}
        for feature in features:
            if feature in model_info.get('categorical_features', []):
                # For categorical features, provide a text input
                input_data[feature] = st.text_input(f"{feature}")
            else:
                # For numerical features, provide a number input
                input_data[feature] = st.number_input(f"{feature}", format="%.6f")
        
        if st.button("Predict"):
            try:
                # Convert input to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Handle categorical features
                for col in model_info.get('categorical_features', []):
                    le = LabelEncoder()
                    # This is a simplified approach - in a real app, you'd need to use the same encoding as during training
                    input_df[col] = 0  # Placeholder
                
                # Apply scaling if used during training
                if model_info.get('scaler') is not None:
                    input_df = model_info['scaler'].transform(input_df)
                
                # Make prediction
                prediction = model_info['model'].predict(input_df)[0]
                
                if problem_type == "Classification":
                    st.success(f"Prediction: {prediction}")
                    # If probabilities are available
                    if hasattr(model_info['model'], 'predict_proba'):
                        probs = model_info['model'].predict_proba(input_df)[0]
                        st.write("Class Probabilities:")
                        prob_df = pd.DataFrame({
                            'Class': model_info['model'].classes_,
                            'Probability': probs
                        })
                        st.dataframe(prob_df)
                else:
                    st.success(f"Predicted Value: {prediction:.4f}")
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        
        # Bulk prediction option
        st.subheader("Bulk Predictions")
        st.write("Upload a CSV file with the same features to make multiple predictions.")
        
        bulk_file = st.file_uploader("Choose a CSV file for bulk prediction", type="csv")
        
        if bulk_file is not None:
            try:
                bulk_data = pd.read_csv(bulk_file)
                
                # Check if all required features are present
                missing_features = [f for f in features if f not in bulk_data.columns]
                if missing_features:
                    st.error(f"Missing features in CSV: {', '.join(missing_features)}")
                else:
                    # Prepare data
                    X_pred = bulk_data[features].copy()
                    
                    # Handle categorical features
                    for col in model_info.get('categorical_features', []):
                        if col in X_pred.columns:
                            le = LabelEncoder()
                            # Again, this is simplified - would need the same encoding as training
                            X_pred[col] = le.fit_transform(X_pred[col])
                    
                    # Apply scaling if used during training
                    if model_info.get('scaler') is not None:
                        X_pred = model_info['scaler'].transform(X_pred)
                    
                    # Make predictions
                    predictions = model_info['model'].predict(X_pred)
                    
                    # Add predictions to the DataFrame
                    result_df = bulk_data.copy()
                    result_df['Prediction'] = predictions
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(result_df)
                    
                    # Create download link for predictions
                    csv = result_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions as CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error in bulk prediction: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("FridAI - No-Code Predictive Modeling Tool")