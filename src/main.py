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

# Set better default styles for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})

# Custom color palettes for attractive visualizations
PALETTE = ["#4361ee", "#3a0ca3", "#7209b7", "#f72585", "#4cc9f0"]
DIVERGING_PALETTE = sns.diverging_palette(230, 20, as_cmap=True)

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

    # Texto explicativo de la aplicación con emojis
    st.markdown(f"""
    ## ✨ Welcome to {APP_NAME}! 🤖

    {APP_NAME} is an interactive tool designed to simplify predictive modeling without writing code.
    Whether you're a data scientist, analyst, or student, this application helps you:

    - 📊 **Upload your data** and get immediate insights
    - 📈 **Explore and visualize** your data through various charts
    - 🧠 **Train powerful machine learning models** with a few clicks
    - 🎛️ **Customize parameters** to improve model performance
    - 💾 **Download your trained model** for use in other applications
    - 🔮 **Make predictions** on new data without coding

    Simply upload your CSV file to get started! 🚀
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
            
            # Create a more visual data information display
            st.subheader("Data Information")
            
            # Create a DataFrame with column information
            info_df = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes.astype(str),
                'Non-Null Count': data.count().values,
                'Null Count': data.isnull().sum().values,
                'Null %': (100 * data.isnull().sum() / len(data)).round(2).astype(str) + '%',
                'Unique Values': [data[col].nunique() for col in data.columns]
            })
            
            # Calculate memory usage for each column
            memory_usage = data.memory_usage(deep=True)
            info_df['Memory Usage'] = [f"{memory_usage[i]/1024:.2f} KB" for i in range(len(data.columns))]
            
            # Format the dataframe with conditional highlighting
            st.dataframe(
                info_df,
                hide_index=True,
                use_container_width=True
            )
            
            # Dataset quick stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{data.shape[0]:,}")
            col2.metric("Total Columns", data.shape[1])
            col3.metric("Missing Cells", f"{data.isnull().sum().sum():,}")
            col4.metric("Memory Usage", f"{data.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
            
            st.subheader("Statistical Summary")
            st.write(data.describe())
            
            # Missing Values section
            st.subheader("Missing Values")
            if data.isnull().sum().sum() > 0:
                missing = data.isnull().sum()
                missing = missing[missing > 0].sort_values(ascending=False)
                
                # Create a bar chart of missing values
                fig, ax = plt.subplots(figsize=(10, 6))
                missing.plot(kind='bar', color="#4361ee", ax=ax)
                plt.title('Missing Values by Column', fontsize=16, pad=20)
                plt.ylabel('Count', fontsize=12)
                plt.xlabel('Columns', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show missing values table
                missing_df = pd.DataFrame({
                    'Column': missing.index,
                    'Missing Count': missing.values,
                    'Missing %': (100 * missing / len(data)).round(2).astype(str) + '%'
                })
                st.dataframe(missing_df, hide_index=True, use_container_width=True)
            else:
                st.success("No missing values found in the dataset!")
                
            # Outlier Detection section
            st.subheader("Outlier Detection")
            
            # Get numeric columns
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_cols:
                # Calculate outliers using IQR method
                outlier_counts = {}
                outlier_percents = {}
                
                for col in numeric_cols:
                    # Skip columns with all missing values
                    if data[col].isna().all():
                        continue
                        
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                    outlier_count = len(outliers)
                    
                    if outlier_count > 0:
                        outlier_counts[col] = outlier_count
                        outlier_percents[col] = (outlier_count / len(data)) * 100
                
                if outlier_counts:
                    # Create DataFrame to display outlier information
                    outlier_df = pd.DataFrame({
                        'Column': outlier_counts.keys(),
                        'Outlier Count': outlier_counts.values(),
                        'Outlier %': [f"{percent:.2f}%" for percent in outlier_percents.values()]
                    }).sort_values('Outlier Count', ascending=False)
                    
                    # Plot outlier counts
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=outlier_df, x='Column', y='Outlier Count', palette='viridis', ax=ax)
                    plt.title('Outliers by Column', fontsize=16, pad=20)
                    plt.ylabel('Count', fontsize=12)
                    plt.xlabel('Columns', fontsize=12)
                    plt.xticks(rotation=45, ha='right')
                    for i, v in enumerate(outlier_df['Outlier Count']):
                        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display outlier table
                    st.dataframe(outlier_df, hide_index=True, use_container_width=True)
                    
                    # Option to visualize outliers in a specific column
                    if len(numeric_cols) > 0:
                        st.subheader("Visualize Outliers")
                        selected_col = st.selectbox("Select column to visualize outliers", 
                                                [col for col in outlier_counts.keys()])
                        
                        # Calculate bounds for selected column
                        Q1 = data[selected_col].quantile(0.25)
                        Q3 = data[selected_col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Create boxplot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.boxplot(x=data[selected_col], ax=ax, color="#4361ee")
                        plt.title(f'Boxplot with Outliers: {selected_col}', fontsize=16, pad=20)
                        
                        # Add annotations for outlier boundaries
                        plt.axvline(x=lower_bound, color='red', linestyle='--', 
                                label=f'Lower bound: {lower_bound:.2f}')
                        plt.axvline(x=upper_bound, color='red', linestyle='--',
                                label=f'Upper bound: {upper_bound:.2f}')
                        plt.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show outlier distribution
                        outlier_data = data[(data[selected_col] < lower_bound) | 
                                        (data[selected_col] > upper_bound)][selected_col]
                        
                        st.write(f"**Outlier Statistics for {selected_col}:**")
                        st.write(f"- Number of outliers: {len(outlier_data)}")
                        st.write(f"- Percentage: {len(outlier_data)/len(data)*100:.2f}%")
                        st.write(f"- Min outlier value: {outlier_data.min()}")
                        st.write(f"- Max outlier value: {outlier_data.max()}")
                else:
                    st.success("No outliers detected in the numeric columns!")
            else:
                st.info("No numeric columns found for outlier detection.")
            
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
                               ["Distribution", "Correlation", "Categorical Analysis", "Scatter Plot"])
        
        if viz_type == "Distribution":
            if numeric_cols:
                col = st.selectbox("Select Numeric Column", numeric_cols)
                
                # Create enhanced distribution plot
                fig, ax = plt.subplots(1, 2, figsize=(PLOT_WIDTH, PLOT_HEIGHT), gridspec_kw={'width_ratios': [2, 1]})
                
                # Histogram with KDE
                sns.histplot(data[col], kde=True, ax=ax[0], color="#4361ee", alpha=0.7)
                ax[0].lines[0].set_color('#f72585')  # Set KDE line color after plotting
                ax[0].lines[0].set_linewidth(2)      # Set KDE line width after plotting
                ax[0].set_title(f'Distribution of {col}', fontsize=14, pad=10)
                ax[0].set_xlabel(col, fontsize=12)
                ax[0].set_ylabel('Frequency', fontsize=12)
                
                # Add mean and median lines
                mean_val = data[col].mean()
                median_val = data[col].median()
                
                ax[0].axvline(mean_val, color='#ff9e00', linestyle='--', linewidth=2, 
                             label=f'Mean: {mean_val:.2f}')
                ax[0].axvline(median_val, color='#38b000', linestyle='-', linewidth=2, 
                             label=f'Median: {median_val:.2f}')
                ax[0].legend(fontsize=10)
                
                # Boxplot
                sns.boxplot(y=data[col], ax=ax[1], palette=["#4361ee"], width=0.4)
                ax[1].set_title(f'Boxplot of {col}', fontsize=14, pad=10)
                ax[1].set_ylabel('')  # Remove y label on boxplot
                
                # Annotate with quartile values
                q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
                ax[1].annotate(f'Q1: {q1:.2f}', xy=(0.1, q1), xytext=(0.4, q1),
                              color='white', fontweight='bold', backgroundcolor='#4361ee', fontsize=9)
                ax[1].annotate(f'Q3: {q3:.2f}', xy=(0.1, q3), xytext=(0.4, q3),
                              color='white', fontweight='bold', backgroundcolor='#4361ee', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistical summary in a nicer format
                st.subheader(f"Statistical Summary for {col}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{data[col].mean():.2f}")
                    st.metric("Min", f"{data[col].min():.2f}")
                with col2:
                    st.metric("Median", f"{data[col].median():.2f}")
                    st.metric("Max", f"{data[col].max():.2f}")
                with col3:
                    st.metric("Std Dev", f"{data[col].std():.2f}")
                    st.metric("Range", f"{data[col].max() - data[col].min():.2f}")
                with col4:
                    st.metric("Skewness", f"{data[col].skew():.2f}")
                    st.metric("Missing", f"{data[col].isna().sum()}")
                
            else:
                st.warning("No numeric columns found in the dataset.")
        
        elif viz_type == "Correlation":
            if len(numeric_cols) > 1:
                st.markdown("### Correlation Heatmap")
                
                # Create a more attractive heatmap
                corr_matrix = data[numeric_cols].corr()
                
                # Allow user to control heatmap size
                heatmap_height = st.slider("Adjust heatmap size", 400, 800, 600)
                
                fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT * (heatmap_height/600)))
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Optional: mask upper triangle
                
                # Create heatmap with annotations
                heatmap = sns.heatmap(
                    corr_matrix, 
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap=DIVERGING_PALETTE,
                    linewidths=1,
                    cbar_kws={"shrink": .8},
                    square=True,
                    ax=ax,
                    vmin=-1, vmax=1,
                    annot_kws={"size": 10}
                )
                
                ax.set_title("Correlation Matrix", fontsize=16, pad=20)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add correlation explanation
                st.markdown("""
                ### Understanding Correlation Values:
                - **1.0**: Perfect positive correlation
                - **0.7 to 0.9**: Strong positive correlation
                - **0.4 to 0.6**: Moderate positive correlation
                - **0.1 to 0.3**: Weak positive correlation
                - **0**: No correlation
                - **-0.1 to -0.3**: Weak negative correlation
                - **-0.4 to -0.6**: Moderate negative correlation
                - **-0.7 to -0.9**: Strong negative correlation
                - **-1.0**: Perfect negative correlation
                """)
                
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
        
        elif viz_type == "Categorical Analysis":
            if categorical_cols:
                cat_col = st.selectbox("Select Categorical Column", categorical_cols)
                
                # Get value counts and sort
                value_counts = data[cat_col].value_counts()
                
                # Handle many categories
                if len(value_counts) > 15:
                    st.info(f"Showing top 15 of {len(value_counts)} categories")
                    value_counts = value_counts.head(15)
                
                # Create enhanced bar plot
                fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
                
                bars = sns.barplot(
                    x=value_counts.index, 
                    y=value_counts.values,
                    palette=sns.color_palette("viridis", len(value_counts)),
                    ax=ax
                )
                
                # Add value labels on top of bars
                for i, v in enumerate(value_counts.values):
                    ax.text(i, v + (value_counts.max() * 0.02), str(v), 
                           ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                ax.set_title(f'Distribution of {cat_col}', fontsize=16, pad=20)
                ax.set_xlabel(cat_col, fontsize=14)
                ax.set_ylabel('Count', fontsize=14)
                
                # Rotate x labels if needed
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Show percentage distribution
                st.subheader("Category Distribution")
                value_percentage = pd.DataFrame({
                    'Count': value_counts.values,
                    'Percentage': (100 * value_counts / value_counts.sum()).round(2)
                })
                value_percentage.index = value_counts.index
                value_percentage['Percentage'] = value_percentage['Percentage'].astype(str) + '%'
                st.dataframe(value_percentage)
                
            else:
                st.warning("No categorical columns found in the dataset.")
                
        elif viz_type == "Scatter Plot":
            if len(numeric_cols) >= 2:
                col1 = st.selectbox("Select X-Axis Column", numeric_cols)
                remaining_cols = [col for col in numeric_cols if col != col1]
                col2 = st.selectbox("Select Y-Axis Column", remaining_cols)
                
                # Optional color by category
                color_by = None
                if categorical_cols:
                    use_color = st.checkbox("Color by category", value=False)
                    if use_color:
                        color_by = st.selectbox("Select category for coloring", categorical_cols)
                
                # Create enhanced scatter plot
                fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
                
                if color_by:
                    # Calculate n_colors based on unique categories
                    n_colors = min(len(data[color_by].unique()), 10)
                    scatter = sns.scatterplot(
                        data=data, 
                        x=col1, 
                        y=col2, 
                        hue=color_by,
                        palette=sns.color_palette("viridis", n_colors),
                        s=80,  # Point size
                        alpha=0.7,
                        ax=ax
                    )
                    
                    # Enhance legend
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=color_by)
                    
                else:
                    scatter = sns.scatterplot(
                        data=data, 
                        x=col1, 
                        y=col2,
                        color="#4361ee",
                        s=80,  # Point size
                        alpha=0.7,
                        ax=ax
                    )
                
                # Add regression line
                sns.regplot(
                    data=data, 
                    x=col1, 
                    y=col2, 
                    scatter=False, 
                    ax=ax,
                    line_kws={"color": "red", "linestyle": "--", "linewidth": 2}
                )
                
                ax.set_title(f'Scatter Plot: {col1} vs {col2}', fontsize=16, pad=20)
                ax.set_xlabel(col1, fontsize=14)
                ax.set_ylabel(col2, fontsize=14)
                
                # Calculate and display correlation
                correlation = data[[col1, col2]].corr().iloc[0, 1]
                ax.annotate(f"Correlation: {correlation:.4f}", 
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="gray", alpha=0.8),
                           fontsize=12)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.warning("Need at least 2 numeric columns for scatter plot.")

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
                            
                            # Get class names
                            if target_encoder:
                                classes = target_encoder.classes_
                                if len(classes) > 10:  # If too many classes, use numeric labels
                                    classes = [str(i) for i in range(len(classes))]
                            else:
                                classes = sorted(np.unique(y_test))
                                if len(classes) > 10:
                                    classes = [str(i) for i in range(len(classes))]
                            
                            # Create enhanced confusion matrix
                            plt.figure(figsize=(10, 8))
                            ax = sns.heatmap(
                                cm, 
                                annot=True, 
                                fmt='d', 
                                cmap='Blues',
                                xticklabels=classes,
                                yticklabels=classes,
                                linewidths=0.5,
                                linecolor='gray',
                                annot_kws={"size": 12, "weight": "bold"},
                                square=True,
                                cbar_kws={"shrink": .8, "label": "Count"}
                            )
                            
                            # Add titles and labels
                            plt.title('Confusion Matrix', fontsize=16, pad=20)
                            plt.xlabel('Predicted Label', fontsize=12)
                            plt.ylabel('True Label', fontsize=12)
                            
                            # Rotate labels if needed
                            if len(classes) > 4:
                                plt.xticks(rotation=45, ha='right')
                            
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            plt.clf()
                            
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
                            
                            # Create scatter plot
                            sns.scatterplot(
                                x=y_test, 
                                y=y_pred, 
                                alpha=0.6, 
                                color="#4361ee",
                                s=80,
                                ax=ax
                            )
                            
                            # Add perfect prediction line
                            min_val = min(y_test.min(), y_pred.min())
                            max_val = max(y_test.max(), y_pred.max())
                            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, 
                                    label='Perfect Prediction')
                            
                            # Add regression line
                            sns.regplot(
                                x=y_test, 
                                y=y_pred, 
                                scatter=False,
                                color='red',
                                line_kws={"linestyle": "-", "linewidth": 2},
                                ax=ax
                            )
                            
                            # Enhance plot appearance
                            ax.set_xlabel('Actual Values', fontsize=14)
                            ax.set_ylabel('Predicted Values', fontsize=14)
                            ax.set_title('Actual vs Predicted Values', fontsize=16, pad=20)
                            
                            # Add metrics annotation
                            ax.annotate(f"MSE: {mse:.4f}\nR²: {r2:.4f}", 
                                       xy=(0.05, 0.95), xycoords='axes fraction',
                                       bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="gray", alpha=0.8),
                                       fontsize=12)
                            
                            ax.legend(loc='lower right')
                            plt.tight_layout()
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
                            
                        # Create feature importance DataFrame
                        feature_imp = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        # Limit to top 20 for readability
                        if len(feature_imp) > 20:
                            feature_imp = feature_imp.head(20)
                        
                        # Create enhanced feature importance plot
                        plt.figure(figsize=(10, 8))
                        ax = sns.barplot(
                            data=feature_imp,
                            y='Feature',
                            x='Importance',
                            palette=sns.color_palette("viridis", len(feature_imp)),
                            edgecolor='black',
                            linewidth=1
                        )
                        
                        # Add values to the bars
                        for i, v in enumerate(feature_imp['Importance']):
                            ax.text(v + 0.01, i, f"{v:.4f}", va='center', fontweight='bold')
                        
                        plt.title('Feature Importance (Top 20)', fontsize=16, pad=20)
                        plt.xlabel('Importance', fontsize=12)
                        plt.ylabel('Feature', fontsize=12)
                        plt.tight_layout()
                        
                        st.pyplot(plt.gcf())
                        plt.clf()
                        
                        # Show feature importance table
                        st.dataframe(feature_imp.reset_index(drop=True))
                
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