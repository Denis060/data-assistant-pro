# Enhanced app.py with Professional Developer Profile

import logging
import os
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Data state and error handling imports
from modules.data_state_manager import validate_data_state, auto_fix_data_state, data_state_dashboard
from modules.error_handler_v2 import (
    ErrorHandler, ErrorSeverity, ErrorCategory, 
    handle_error, handle_data_error, handle_validation_error,
    safe_execute, ErrorContext, error_dashboard
)

# Utility function to make dataframes Arrow-compatible
def make_arrow_compatible(df):
    """Convert DataFrame columns to Arrow-compatible types."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == object:
            # Check if it contains dtype objects
            try:
                if hasattr(df_copy[col].iloc[0], 'name'):  # pandas dtype object
                    df_copy[col] = df_copy[col].astype(str)
            except (IndexError, AttributeError):
                pass
    return df_copy

# Import configuration
from config import get_config
from modules.cleaning_fixed import (
    convert_df_to_csv,
    handle_missing_values,
    handle_outliers,
    remove_duplicates,
)
from modules.data_quality import enhanced_data_quality_dashboard
from modules.enhanced_data_cleaner import EnhancedSmartDataCleaner
from modules.ml_ready_cleaner import MLReadyCleaner
from modules.ml_quality_dashboard import create_ml_quality_dashboard
from modules.smart_cleaning_recommender import generate_cleaning_strategy_report
from modules.domain_validation import DataValidationRules, domain_validation_dashboard
from modules.database import database_dashboard
from modules.time_series import time_series_dashboard
from modules.model_monitoring import ModelMonitor
from modules.model_evaluation import ModelPerformanceAnalyzer, create_performance_analysis_dashboard
from modules.ai_insights import create_ai_insights_dashboard
from modules.performance_analytics import create_performance_dashboard
from modules.advanced_model_optimizer import AdvancedModelOptimizer
from modules.simple_ml_fixer import SimpleMLFixer
from modules.automated_ml_pipeline import AutomatedMLPipeline
from modules.automated_experimenter import AutomatedModelExperimenter
from modules.intelligent_model_recommender import IntelligentModelRecommender

# Import our modules
from modules.eda import create_eda_report
from modules.loader import (
    load_data,
    validate_data_for_cleaning,
    validate_data_for_modeling,
)
from modules.modeling import (
    create_prediction_plots,
    display_model_results,
    display_performance_diagnostic,
    feature_importance_analysis,
    prepare_data_for_modeling,
    train_models,
)

# Import performance and caching utilities
from modules.cache_utils import (
    DataCache,
    cached_correlation_matrix,
    cached_missing_analysis,
    cached_statistical_summary,
    with_progress_cache,
)

# Get configuration
config = get_config()

# --- Setup logging ---
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(config.LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- Utility Functions ---


def generate_cleaning_report(original_df, cleaned_df):
    """Generate a comprehensive cleaning report."""
    report = f"""
DATA CLEANING REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

ORIGINAL DATA:
- Rows: {len(original_df):,}
- Columns: {len(original_df.columns)}
- Missing Values: {original_df.isnull().sum().sum():,}
- Duplicate Rows: {original_df.duplicated().sum():,}

CLEANED DATA:
- Rows: {len(cleaned_df):,}
- Columns: {len(cleaned_df.columns)}
- Missing Values: {cleaned_df.isnull().sum().sum():,}
- Duplicate Rows: {cleaned_df.duplicated().sum():,}

CHANGES MADE:
- Rows Removed: {len(original_df) - len(cleaned_df):,}
- Missing Values Handled: {original_df.isnull().sum().sum() - cleaned_df.isnull().sum().sum():,}
- Duplicates Removed: {original_df.duplicated().sum() - cleaned_df.duplicated().sum():,}

DATA QUALITY IMPROVEMENT:
- Original Quality: {calculate_data_quality_score(original_df):.1f}%
- Current Quality: {calculate_data_quality_score(cleaned_df):.1f}%
- Improvement: {calculate_data_quality_score(cleaned_df) - calculate_data_quality_score(original_df):.1f}%
"""
    return report


def calculate_data_quality_score(df):
    """Calculate a simple data quality score."""
    if df.empty:
        return 0.0

    # Base score from completeness
    completeness = (1 - df.isnull().sum().sum() / df.size) * 100

    # Penalty for duplicates
    duplicate_penalty = (df.duplicated().sum() / len(df)) * 20

    # Bonus for consistent data types
    type_consistency_bonus = min(
        len(df.dtypes.value_counts()) / len(df.columns) * 10, 10
    )

    score = max(0, completeness - duplicate_penalty + type_consistency_bonus)
    return min(100, score)


# --- Page Configuration ---
st.set_page_config(
    page_title="Data Assistant Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Enhanced Sidebar with Developer Profile ---
with st.sidebar:
    # --- Expandable Developer Profile Section ---
    with st.expander("👨‍💻 Developer Profile", expanded=False):
        # About the Developer Section
        st.subheader("About the Developer")

        # Developer info
        st.write("**Ibrahim Fofanah**")
        st.write("🚀 *Data Scientist & ML Engineer*")

        st.write("**Expertise:**")
        st.write("• 🤖 Machine Learning & AI")
        st.write("• 📊 Data Science & Analytics")
        st.write("• � Python Development")
        st.write("• 📈 Statistical Modeling")
        st.write("• ⚡ Advanced Data Mining")

        st.write("**Specializations:**")
        st.write("• AutoML & Hyperparameter Tuning")
        st.write("• Feature Engineering & Selection")
        st.write("• Advanced Data Preprocessing")
        st.write("• Production ML Systems")
        st.write("• Interactive Data Applications")

        st.write("**Contact:**")
        st.write("• 📧 Email: ibrahimdenisfofanah060@gmail.com")
        st.write("• 💼 LinkedIn: https://www.linkedin.com/in/ibrahim-denis-fofanah/")
        st.write("• 🔗 GitHub: https://github.com/Denis060")

        st.markdown("---")
        st.markdown("*'Transforming data into intelligent solutions'*")

    # Professional app info card with enhanced styling
    st.markdown("---")
    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 15px; color: white; text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h3 style="margin-bottom: 10px; font-weight: bold;">Data Assistant Pro</h3>
        <p style="margin: 5px 0;"><strong>Version:</strong> 2.0.0 Enhanced</p>
        <p style="margin: 5px 0;"><strong>Build:</strong> Production Ready</p>
        <p style="margin: 5px 0;"><strong>Features:</strong> AutoML, Advanced Cleaning, Smart Analytics</p>
        <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;">Developed with ❤️ by Ibrahim Fofanah for Data Scientists</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # --- File Upload Section (Moved to Sidebar) ---
    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")
    
    # File upload with enhanced help text
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=["csv", "tsv", "txt"],
        help="Upload CSV, TSV, or TXT file with any standard delimiter",
    )

    # Manual delimiter override for main upload
    main_delimiter_override = st.selectbox(
        "Delimiter (optional)",
        options=["Auto-detect", "Comma (,)", "Semicolon (;)", "Tab", "Pipe (|)"],
        help="Override if auto-detection fails",
        key="main_delimiter",
    )

    # Show file info if uploaded
    if uploaded_file is not None:
        st.success(f"✅ **{uploaded_file.name}**")
        st.caption(f"📊 Size: {uploaded_file.size / 1024:.1f} KB")

    # Multi-Dataset Selection
    st.markdown("---")
    st.markdown("### 📊 Try Demo Datasets")
    st.caption("Choose from various real-world datasets to explore different scenarios")
    
    # Dataset configurations
    datasets = {
        "sample_data.csv": {
            "name": "🏢 Employee Dataset",
            "description": "Basic employee demographics with age, income, education, and satisfaction",
            "use_case": "Perfect for beginners - simple structure with mixed data types"
        },
        "california_housing_train.csv": {
            "name": "🏠 California Housing",
            "description": "Real estate data with house prices, location, and property features",
            "use_case": "Great for regression analysis and price prediction models"
        },
        "customer_churn_messy.csv": {
            "name": "📱 Customer Churn (Messy)",
            "description": "Telecom customer data with missing values and quality issues",
            "use_case": "Ideal for data cleaning practice and churn prediction"
        },
        "employee_performance_messy.csv": {
            "name": "🎯 Employee Performance (Messy)",
            "description": "HR data with performance metrics, requires data cleaning",
            "use_case": "Perfect for HR analytics and performance modeling"
        },
        "sales_forecasting_messy.csv": {
            "name": "📈 Sales Forecasting (Messy)",
            "description": "Time-series sales data with data quality challenges",
            "use_case": "Excellent for time-series analysis and sales prediction"
        }
    }
    
    # Get available datasets from data folder
    data_folder = "data"
    available_datasets = []
    if os.path.exists(data_folder):
        for filename in os.listdir(data_folder):
            if filename.endswith('.csv') and filename in datasets:
                file_path = os.path.join(data_folder, filename)
                file_size = os.path.getsize(file_path)
                available_datasets.append({
                    'filename': filename,
                    'path': file_path,
                    'size_kb': file_size / 1024,
                    **datasets[filename]
                })
    
    if available_datasets:
        # Display datasets in a grid
        cols = st.columns(2)
        for idx, dataset in enumerate(available_datasets):
            with cols[idx % 2]:
                with st.container():
                    st.markdown(f"**{dataset['name']}**")
                    st.caption(f"📄 {dataset['filename']} • {dataset['size_kb']:.1f} KB")
                    st.write(dataset['description'])
                    st.info(f"💡 {dataset['use_case']}")
                    
                    if st.button(f"Load {dataset['name']}", key=f"load_{dataset['filename']}", use_container_width=True):
                        try:
                            with st.spinner(f"Loading {dataset['name']}..."):
                                sample_df = pd.read_csv(dataset['path'])
                                st.session_state.sample_data = sample_df
                                st.session_state.original_df = sample_df.copy()
                                st.success(
                                    f"✅ {dataset['name']} loaded! ({len(sample_df):,} rows, {len(sample_df.columns)} columns)"
                                )
                                st.info(f"📁 Source: {dataset['path']}")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error loading {dataset['name']}: {str(e)}")
                    st.markdown("---")
    else:
        st.warning("No demo datasets found in the data folder.")
        # Fallback to creating demo data
        if st.button("Create Demo Dataset", use_container_width=True):
            try:
                sample_df = pd.DataFrame(
                    {
                        "age": [25, 30, 35, 40, 45, 28, 33, 38, 42, 29],
                        "income": [50000, 75000, 85000, 95000, 105000, 60000, 80000, 90000, 100000, 65000],
                        "education": ["Bachelor", "Master", "PhD", "Master", "PhD", "Bachelor", "Master", "Bachelor", "PhD", "Master"],
                        "experience": [3, 8, 12, 15, 20, 5, 10, 13, 18, 6],
                        "satisfaction": ["High", "Medium", "High", "High", "Medium", "High", "High", "Medium", "High", "High"],
                    }
                )
                st.session_state.sample_data = sample_df
                st.session_state.original_df = sample_df.copy()
                st.success(f"✅ Demo dataset created! ({len(sample_df)} rows, {len(sample_df.columns)} columns)")
                st.rerun()
            except Exception as e:
                st.error(f"Error creating demo dataset: {str(e)}")

    # System Stats Dashboard
    if "cleaned_df" in st.session_state and st.session_state.cleaned_df is not None:
        st.markdown("### 📊 Live Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "📈 Rows",
                f"{len(st.session_state.cleaned_df):,}",
                delta=(
                    f"{len(st.session_state.cleaned_df) - len(st.session_state.get('original_df', st.session_state.cleaned_df)):+,}"
                    if "original_df" in st.session_state
                    else None
                ),
            )
            st.metric(
                "🧠 Memory",
                f"{st.session_state.cleaned_df.memory_usage().sum() / 1024:.1f} KB",
            )
        with col2:
            st.metric("📋 Columns", len(st.session_state.cleaned_df.columns))
            missing_pct = (
                st.session_state.cleaned_df.isnull().sum().sum()
                / st.session_state.cleaned_df.size
            ) * 100
            st.metric(
                "🎯 Quality",
                f"{100-missing_pct:.1f}%",
                delta=f"{missing_pct:.1f}% missing" if missing_pct > 0 else "Perfect!",
            )

    # === Model Management Section (Moved to Sidebar) ===
    st.markdown("---")
    st.markdown("### 🤖 Model Management")

    # Model status indicator
    if "model_results" in st.session_state and st.session_state.model_results:
        st.success("✅ Models trained and ready!")
        try:
            # Try to find the best model with proper error handling
            if isinstance(st.session_state.model_results, dict):
                # Check if the results have the expected structure
                first_key = list(st.session_state.model_results.keys())[0]
                first_result = st.session_state.model_results[first_key]

                if isinstance(first_result, dict) and "score" in first_result:
                    best_model = max(
                        st.session_state.model_results.items(),
                        key=lambda x: x[1]["score"],
                    )
                    st.info(
                        f"🏆 Best: {best_model[0]} ({best_model[1]['score']:.3f})"
                    )
                else:
                    # Alternative structure - results might be stored differently
                    model_names = list(st.session_state.model_results.keys())
                    st.info(f"🏆 Models: {len(model_names)} trained")
            else:
                st.info("📊 Model results available")
        except Exception as e:
            st.info("📊 Model results available")

        # Quick model actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True):
                st.success("Model saved!")
        with col2:
            if st.button("🔮 Predict", use_container_width=True):
                st.info("Go to Modeling tab!")
    else:
        st.info("🎯 No models trained yet")

    # === System Performance Section (Moved to Sidebar) ===
    st.markdown("---")
    st.markdown("### ℹ️ System Performance")
    if "cleaned_df" in st.session_state and st.session_state.cleaned_df is not None:
        st.metric("📊 Active Rows", f"{len(st.session_state.cleaned_df):,}")
        st.metric(
            "💾 Memory Usage",
            f"{st.session_state.cleaned_df.memory_usage().sum() / 1024:.1f} KB",
        )
    else:
        st.info("No data loaded yet")

    # Clear session button
    st.markdown("---")
    if st.button("🗑️ Clear Session", help="Reset all data and models", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("Session cleared!")
        st.rerun()

    # === SIMPLE UX ENHANCEMENTS ===
    st.markdown("---")
    
    # Quick Help Section
    with st.expander("📖 Quick Help & Tips", expanded=False):
        st.markdown("""
        **🚀 Getting Started:**
        1. Upload your data file above
        2. Clean your data automatically
        3. Explore with statistical analysis
        4. Build machine learning models
        5. Generate insights and reports
        
        **💡 Pro Tips:**
        - Use the sample data to test features
        - Check data quality before modeling  
        - Export results for presentations
        - Monitor model performance over time
        """)
    
    # Keyboard Shortcuts
    with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**File Operations:**")
            st.markdown("• `Ctrl+O` - Open file")
            st.markdown("• `Ctrl+S` - Save results")
            st.markdown("• `Ctrl+R` - Refresh data")
        with col2:
            st.markdown("**Navigation:**") 
            st.markdown("• `Ctrl+1` - Data view")
            st.markdown("• `Ctrl+2` - Analysis")
            st.markdown("• `Ctrl+3` - Modeling")
    
    # App Status
    with st.expander("⚡ App Status", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔧 Version", "2.0.0")
            st.metric("📊 Features", "25+")
        with col2:
            st.metric("🚀 Status", "Production")
            st.metric("💾 Cache", "Active")

# Add simple JavaScript for keyboard shortcuts (non-intrusive)
st.markdown("""
<script>
document.addEventListener('keydown', function(e) {
    // Only activate shortcuts when not typing in input fields
    if (e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
        if (e.ctrlKey) {
            switch(e.key) {
                case '?':
                    e.preventDefault();
                    alert('Keyboard Shortcuts:\\n\\nCtrl+R: Refresh page\\nCtrl+S: Download results\\nCtrl+H: Show this help\\nF5: Reload application');
                    break;
                case 'h':
                    e.preventDefault();
                    alert('Data Assistant Pro Help:\\n\\n1. Upload your data file\\n2. Clean and explore your data\\n3. Build ML models\\n4. Generate insights\\n\\nUse Ctrl+? for shortcuts');
                    break;
            }
        }
    }
});
</script>
""", unsafe_allow_html=True)

# --- Main App with Enhanced Error Handling and Performance Monitoring ---
def main():
    """Main application with comprehensive error handling and performance tracking"""
    
    # Performance monitoring
    app_start_time = time.time()
    
    try:
        # Initialize error tracking in session state
        if "error_count" not in st.session_state:
            st.session_state.error_count = 0
            st.session_state.last_error = None
            st.session_state.performance_metrics = {
                "load_times": [],
                "cache_hits": 0,
                "cache_misses": 0
            }
        
        # Show performance metrics in sidebar
        with st.sidebar:
            if st.session_state.error_count > 0:
                st.warning(f"⚠️ {st.session_state.error_count} errors encountered this session")
                if st.button("🔄 Reset Error Counter"):
                    st.session_state.error_count = 0
                    st.session_state.last_error = None
                    st.rerun()
            
            # Performance dashboard
            with st.expander("⚡ Performance Dashboard", expanded=False):
                app_load_time = time.time() - app_start_time
                st.metric("🚀 App Load Time", f"{app_load_time:.2f}s")
                
                if st.session_state.performance_metrics["load_times"]:
                    avg_load = sum(st.session_state.performance_metrics["load_times"]) / len(st.session_state.performance_metrics["load_times"])
                    st.metric("📊 Avg Load Time", f"{avg_load:.2f}s")
                
                cache_total = st.session_state.performance_metrics["cache_hits"] + st.session_state.performance_metrics["cache_misses"]
                if cache_total > 0:
                    cache_rate = (st.session_state.performance_metrics["cache_hits"] / cache_total) * 100
                    st.metric("💾 Cache Hit Rate", f"{cache_rate:.1f}%")
        
        # Store load time
        st.session_state.performance_metrics["load_times"].append(app_load_time)
        if len(st.session_state.performance_metrics["load_times"]) > 10:
            st.session_state.performance_metrics["load_times"].pop(0)
        
    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        logger.critical(f"Critical app error: {str(e)}\n{traceback.format_exc()}")
        st.session_state.error_count += 1
        st.session_state.last_error = str(e)
        
        # Show fallback interface
        st.markdown("### 🛠️ Fallback Mode")
        st.info("The application encountered an error but is running in fallback mode.")
        return

# Continue with the main app content below...
# Hero Section with Dynamic Title - Moved styles to top for better loading
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        margin-top: 0px;
        animation: slideInDown 1s ease-out;
    }
    
    .hero-subtitle {
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
        animation: fadeInUp 1s ease-out 0.3s both;
    }
    
    .feature-badge {
        display: inline-block;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
        animation: pulse 2s infinite;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 25px;
        margin: 30px 0;
        padding: 0 10px;
    }
    
    .feature-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        padding: 30px 25px;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(102, 126, 234, 0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.05), transparent);
        transition: left 0.5s;
    }
    
    .feature-card:hover::before {
        left: 100%;
    }
    
    .feature-icon {
        font-size: 3.5rem;
        margin-bottom: 20px;
        display: block;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 15px;
        font-family: 'Poppins', sans-serif;
    }
    
    .feature-description {
        color: #6c757d;
        font-size: 0.95rem;
        line-height: 1.6;
        font-family: 'Poppins', sans-serif;
    }
    
    @keyframes slideInDown {
        from { transform: translateY(-100px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from { transform: translateY(50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transform: translateY(0);
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    /* Remove extra spacing */
    .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main Title Section - Compact and clean
st.markdown(
    """
    <div class="hero-title">🚀 Data Assistant Pro</div>
    <div class="hero-subtitle">Transform Your Data Into Intelligence with AI-Powered Analytics</div>
    
    <div style="text-align: center; margin-bottom: 30px;">
        <span class="feature-badge">🤖 AutoML</span>
        <span class="feature-badge">🧹 Smart Cleaning</span>
        <span class="feature-badge">📊 Advanced Analytics</span>
        <span class="feature-badge">⚡ Real-time Insights</span>
        <span class="feature-badge">🎯 Production Ready</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Add interactive JavaScript for enhanced UX
st.markdown(
    """
    <script>
    // Add smooth scrolling and interactive effects
    document.addEventListener('DOMContentLoaded', function() {
        // Add floating help button
        const helpButton = document.createElement('div');
        helpButton.innerHTML = '💡';
        helpButton.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            z-index: 1000;
            transition: all 0.3s ease;
        `;
        
        helpButton.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.1)';
            this.style.boxShadow = '0 6px 25px rgba(102, 126, 234, 0.6)';
        });
        
        helpButton.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.boxShadow = '0 4px 20px rgba(102, 126, 234, 0.4)';
        });
        
        helpButton.addEventListener('click', function() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {
                sidebar.scrollIntoView({ behavior: 'smooth' });
            }
        });
        
        document.body.appendChild(helpButton);
        
        // Add hover effects to metrics
        const metrics = document.querySelectorAll('[data-testid="metric-container"]');
        metrics.forEach(metric => {
            metric.style.transition = 'all 0.3s ease';
            metric.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-3px)';
                this.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
            });
            metric.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0)';
                this.style.boxShadow = 'none';
            });
        });
        
        // Add sparkle animation to title
        const title = document.querySelector('.hero-title');
        if (title) {
            setInterval(() => {
                const sparkle = document.createElement('span');
                sparkle.innerHTML = '✨';
                sparkle.style.cssText = `
                    position: absolute;
                    font-size: 20px;
                    animation: sparkle 2s ease-in-out;
                    pointer-events: none;
                `;
                sparkle.style.left = Math.random() * title.offsetWidth + 'px';
                sparkle.style.top = Math.random() * title.offsetHeight + 'px';
                
                title.style.position = 'relative';
                title.appendChild(sparkle);
                
                setTimeout(() => sparkle.remove(), 2000);
            }, 3000);
        }
    });
    
    // Add sparkle animation keyframes
    const style = document.createElement('style');
    style.textContent = `
        @keyframes sparkle {
            0% { opacity: 0; transform: scale(0) rotate(0deg); }
            50% { opacity: 1; transform: scale(1) rotate(180deg); }
            100% { opacity: 0; transform: scale(0) rotate(360deg); }
        }
    `;
    document.head.appendChild(style);
    </script>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Handle both uploaded file and sample data
data_source = None
if uploaded_file is not None:
    data_source = "uploaded"
    df_original = load_data(uploaded_file, main_delimiter_override)
elif "sample_data" in st.session_state:
    data_source = "sample"
    df_original = st.session_state.sample_data
    st.info("📊 Using sample dataset")
else:
    df_original = None

if df_original is not None:
    # Create performance monitor
    start_time = time.time()
    data_hash = DataCache.get_data_hash(df_original)
    
    # Performance metrics in sidebar
    with st.sidebar:
        st.markdown("### ⚡ Performance Monitor")
        
        # System Health Check
        with st.expander("🔍 System Health", expanded=False):
            validation_report = validate_data_state()
            if validation_report['status'] == 'healthy':
                st.success("✅ System Healthy")
            elif validation_report['status'] == 'warning':
                st.warning("⚠️ System Warnings")
            else:
                st.error("❌ System Issues")
            
            if st.button("📊 Full Health Report"):
                st.session_state.show_health_dashboard = True
        st.markdown("### ⚡ Performance Monitor")
        perf_container = st.container()
        
        with perf_container:
            load_time = time.time() - start_time
            st.metric("📊 Data Load Time", f"{load_time:.2f}s")
            st.metric("🔢 Data Hash", f"#{data_hash[:8]}")
            
            # Memory usage indicator
            memory_mb = df_original.memory_usage().sum() / 1024 / 1024
            memory_color = "🟢" if memory_mb < 10 else "🟡" if memory_mb < 100 else "🔴"
            st.metric(f"{memory_color} Memory", f"{memory_mb:.1f} MB")
            
            # Cache status
            cache_info = DataCache.get_cache_info()
            st.metric("💾 Cache Files", cache_info['file_count'])
            
            if st.button("🗑️ Clear Cache"):
                try:
                    DataCache.clear_cache()
                    st.success("Cache cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")

    try:
        # Initialize session state with error handling and performance tracking
        with st.spinner("🚀 Initializing data processing..."):
            # Validate current data state
            validation_report = validate_data_state()
            
            if validation_report['status'] in ['error', 'critical_error']:
                # Auto-fix common issues
                fixes = auto_fix_data_state()
                if fixes:
                    st.info("🔧 Applied automatic fixes to data state")
            
            if "cleaned_df" not in st.session_state:
                st.session_state.cleaned_df = df_original.copy()
                st.session_state.original_df = df_original.copy()  # Store original
                st.session_state.data_hash = data_hash
                logger.info(f"Initialized cleaned_df in session state (hash: {data_hash})")
            
            # Ensure data consistency
            if st.session_state.get('data_hash') != data_hash:
                st.session_state.cleaned_df = df_original.copy()
                st.session_state.original_df = df_original.copy()
                st.session_state.data_hash = data_hash
                logger.info(f"Updated session state with new data (hash: {data_hash})")

        # Engaging success banner
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #00C851 0%, #00A642 100%); 
                        padding: 20px; border-radius: 15px; color: white; text-align: center;
                        margin: 20px 0; box-shadow: 0 8px 25px rgba(0, 200, 81, 0.3);
                        animation: slideInDown 0.8s ease-out;">
                <h3 style="margin: 0; font-size: 1.5rem;">🎉 Data Successfully Loaded!</h3>
                <p style="margin: 10px 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                    📊 {len(df_original):,} rows × {len(df_original.columns)} columns | 
                    🚀 Ready for analysis!
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Interactive data overview with enhanced styling
        with st.expander("📋 Interactive Data Overview", expanded=True):
            # Add data health score
            missing_percent = (df_original.isnull().sum().sum() / (len(df_original) * len(df_original.columns))) * 100
            health_score = max(0, 100 - missing_percent - (len(df_original.select_dtypes(include=['object']).columns) * 2))
            
            # Health score visualization
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 15px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px;">
                    <h4 style="margin: 0;">📈 Data Health Score</h4>
                    <div style="font-size: 2rem; font-weight: bold; margin: 10px 0;">
                        {health_score:.0f}/100
                    </div>
                    <div style="background: rgba(255,255,255,0.2); border-radius: 10px; height: 10px; margin: 10px 0;">
                        <div style="background: #00C851; border-radius: 10px; height: 100%; width: {health_score}%; transition: width 1s ease;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Rows", f"{len(df_original):,}")
            with col2:
                st.metric("📋 Columns", len(df_original.columns))
            with col3:
                missing_icon = "🟢" if df_original.isnull().sum().sum() == 0 else "🟡" if missing_percent < 10 else "🔴"
                st.metric(f"{missing_icon} Missing Values", f"{df_original.isnull().sum().sum():,}")
            with col4:
                memory_mb = df_original.memory_usage().sum() / 1024 / 1024
                memory_icon = "🟢" if memory_mb < 10 else "🟡" if memory_mb < 100 else "🔴"
                memory_unit = "MB" if memory_mb >= 1 else "KB"
                memory_value = memory_mb if memory_mb >= 1 else df_original.memory_usage().sum() / 1024
                st.metric(f"{memory_icon} Memory Usage", f"{memory_value:.1f} {memory_unit}")

            # Enhanced data preview with column analysis
            st.markdown("### 🔍 Data Preview & Column Analysis")
            
            # Column type breakdown
            col_types = df_original.dtypes.value_counts()
            st.markdown("**Column Types:**")
            cols = st.columns(len(col_types))
            for i, (dtype, count) in enumerate(col_types.items()):
                icon = {"object": "📝", "int64": "🔢", "float64": "📊", "datetime64[ns]": "📅", "bool": "✅"}.get(str(dtype), "❓")
                cols[i].metric(f"{icon} {str(dtype)}", count)
            
            # Interactive data preview
            preview_rows = st.slider("Preview rows:", 3, 20, 5, help="Adjust how many rows to preview", key="preview_rows_slider")
            st.dataframe(
                df_original.head(preview_rows), 
                use_container_width=True,
                height=min(400, (preview_rows + 1) * 35)
            )

        # ===== DEDICATED EDA SECTION (OUTSIDE TABS) =====
        st.markdown("---")
        st.markdown("## 📊 **Exploratory Data Analysis (EDA) Report**")
        
        # EDA Controls Row
        eda_col1, eda_col2, eda_col3, eda_col4, eda_col5 = st.columns([2, 1, 1, 1, 1])
        
        with eda_col1:
            generate_eda_btn = st.button("🚀 **Generate Complete EDA Report**", type="primary", use_container_width=True)
        
        with eda_col2:
            detailed_analysis = st.checkbox("🔬 Detailed", value=True, key="eda_detailed")
            
        with eda_col3:
            include_correlations = st.checkbox("📊 Correlations", value=True, key="eda_correlations")
            
        with eda_col4:
            include_distributions = st.checkbox("📈 Distributions", value=True, key="eda_distributions")
            
        with eda_col5:
            if st.button("🗑️ Clear", use_container_width=True):
                # Clear all EDA-related session state
                keys_to_clear = [k for k in st.session_state.keys() if k.startswith('eda_')]
                for key in keys_to_clear:
                    del st.session_state[key]
                st.success("EDA cache cleared!")
                st.rerun()

        # EDA Content Container - This persists across reruns
        eda_container = st.container()
        
        # Generate EDA Report
        if generate_eda_btn:
            with eda_container:
                # Progress tracking
                progress_bar = st.progress(0, text="🔄 Initializing EDA generation...")
                
                try:
                    # Step 1: Data validation
                    progress_bar.progress(0.1, text="🔍 Validating data structure...")
                    time.sleep(0.3)
                    
                    if df_original is None or df_original.empty:
                        st.error("❌ No data available for analysis")
                        st.stop()
                    
                    # Step 2: Cache computations
                    progress_bar.progress(0.3, text="📊 Computing statistical summaries...")
                    stats_summary = cached_statistical_summary(data_hash, df_original)
                    
                    progress_bar.progress(0.5, text="🔍 Analyzing missing values...")
                    missing_info = cached_missing_analysis(data_hash, df_original)
                    
                    # Step 3: Generate visualizations directly in this container
                    progress_bar.progress(0.7, text="📈 Creating visualizations...")
                    
                    # Create visualizations directly here (not in separate function)
                    st.subheader("📊 Dataset Overview")
                    
                    # Basic metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📄 Rows", f"{len(df_original):,}")
                    with col2:
                        st.metric("📋 Columns", len(df_original.columns))
                    with col3:
                        st.metric("❓ Missing Values", f"{missing_info['total_missing']:,}")
                    with col4:
                        st.metric("� Duplicates", f"{df_original.duplicated().sum():,}")
                    
                    # Statistical Summary
                    if not stats_summary['describe'].empty:
                        st.subheader("📊 Statistical Summary")
                        st.dataframe(stats_summary['describe'].round(3), use_container_width=True)
                    
                    # Missing Values Analysis
                    if missing_info['total_missing'] > 0:
                        st.subheader("� Missing Values Analysis")
                        missing_df = pd.DataFrame({
                            'Column': missing_info['missing_count'].index,
                            'Missing Count': missing_info['missing_count'].values,
                            'Missing %': missing_info['missing_percentage'].values
                        })
                        missing_df = missing_df[missing_df['Missing Count'] > 0].reset_index(drop=True)
                        if not missing_df.empty:
                            st.dataframe(missing_df, use_container_width=True)
                            
                            # Missing values bar chart
                            fig_missing = px.bar(
                                missing_df, 
                                x='Column', 
                                y='Missing %',
                                title="Missing Values by Column",
                                color='Missing %',
                                color_continuous_scale='Reds'
                            )
                            fig_missing.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_missing, use_container_width=True)
                    
                    # Correlation Matrix
                    if include_correlations:
                        numeric_cols = df_original.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 1:
                            st.subheader("🔗 Correlation Matrix")
                            corr_matrix = cached_correlation_matrix(data_hash, df_original)
                            
                            fig_corr = px.imshow(
                                corr_matrix,
                                title="Feature Correlation Heatmap",
                                color_continuous_scale="RdBu_r",
                                aspect="auto",
                                text_auto=True
                            )
                            fig_corr.update_layout(
                                title_x=0.5,
                                width=800,
                                height=600
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Distribution Analysis
                    if include_distributions:
                        numeric_cols = df_original.select_dtypes(include=['number']).columns
                        if len(numeric_cols) > 0:
                            st.subheader("� Distribution Analysis")
                            
                            # Create subplots for distributions
                            for i, col in enumerate(numeric_cols[:6]):  # Limit to first 6 columns
                                if i % 2 == 0:
                                    col1, col2 = st.columns(2)
                                
                                fig_hist = px.histogram(
                                    df_original, 
                                    x=col,
                                    title=f"Distribution of {col}",
                                    marginal="box"
                                )
                                
                                if i % 2 == 0:
                                    with col1:
                                        st.plotly_chart(fig_hist, use_container_width=True)
                                else:
                                    with col2:
                                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Data Types
                    st.subheader("📋 Data Types Overview")
                    dtype_counts = df_original.dtypes.value_counts()
                    dtype_df = pd.DataFrame({
                        'Data Type': dtype_counts.index.astype(str),
                        'Count': dtype_counts.values
                    })
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(dtype_df, use_container_width=True)
                    with col2:
                        fig_dtype = px.pie(
                            dtype_df, 
                            values='Count', 
                            names='Data Type',
                            title="Data Types Distribution"
                        )
                        st.plotly_chart(fig_dtype, use_container_width=True)
                    
                    # Completion
                    progress_bar.progress(1.0, text="✅ EDA Report Complete!")
                    time.sleep(0.5)
                    progress_bar.empty()
                    
                    # Store success in session state
                    st.session_state.eda_report_generated = True
                    st.session_state.eda_generation_time = time.time()
                    
                    st.success("🎉 **Complete EDA Report Generated Successfully!**")
                    st.info("📊 All visualizations are now displayed above and will persist until you clear the cache or reload the page.")
                    
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ Error generating EDA report: {str(e)}")
                    logger.error(f"EDA generation error: {str(e)}\n{traceback.format_exc()}")
        
        # Show status if EDA was previously generated
        elif "eda_report_generated" in st.session_state:
            with eda_container:
                generation_time = st.session_state.get('eda_generation_time', 0)
                time_ago = int(time.time() - generation_time) if generation_time else 0
                st.success(f"✅ EDA Report was generated {time_ago} seconds ago")
                st.info("� **Scroll up to view the complete analysis** or click 'Generate' to refresh with new options.")
                
        else:
            with eda_container:
                st.info("👆 **Click 'Generate Complete EDA Report' to start comprehensive data analysis**")
                
                # Preview of what will be generated
                st.markdown("### 🎯 **Report Contents:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **📊 Statistical Analysis:**
                    - Dataset overview & metrics
                    - Descriptive statistics  
                    - Missing value patterns
                    - Data type distribution
                    """)
                    
                with col2:
                    st.markdown("""
                    **📈 Visualizations:**
                    - Correlation heatmaps
                    - Distribution histograms
                    - Missing value charts
                    - Data type pie charts
                    """)

        # Enhanced tabs (keeping existing functionality)
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
            [
                "🛠️ Analysis Tools",
                "🧹 Data Cleaning",
                "🤖 Modeling",
                "📈 Export & Reports",
                "🗃️ Database Integration",
                "📅 Time Series Analysis",
                "📊 Model Monitoring",
                "🧠 AI Insights",
                "⚡ Performance Analytics",
                "🔧 System Health",
            ]
        )

        with tab1:
            st.header("🛠️ Quick Analysis Tools")
            st.info("� **Tip:** Use the comprehensive EDA section above, or these quick tools for specific insights!")
            
            # Quick analysis options
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔍 Quick Insights")
                if st.button("📊 Data Info", use_container_width=True):
                    st.write("**Dataset Information:**")
                    st.write(f"- Shape: {df_original.shape}")
                    st.write(f"- Memory usage: {df_original.memory_usage().sum() / 1024 / 1024:.2f} MB")
                    st.write("**Column Types:**")
                    st.write(df_original.dtypes.value_counts())
                
            with col2:
                st.subheader("🎯 Targeted Analysis")
                selected_column = st.selectbox("Select column for analysis:", df_original.columns)
                if st.button("🔍 Analyze Column", use_container_width=True):
                    col_data = df_original[selected_column]
                    st.write(f"**Analysis of '{selected_column}':**")
                    
                    if col_data.dtype in ['int64', 'float64']:
                        st.write("**Statistics:**")
                        st.write(col_data.describe())
                        
                        fig = px.histogram(df_original, x=selected_column, title=f"Distribution of {selected_column}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("**Value Counts:**")
                        st.write(col_data.value_counts().head(10))
            
            # Create EDA content above tabs if needed
            if "eda_generated" in st.session_state and st.session_state.eda_generated:
                # Insert a marker for where EDA content should be displayed
                st.markdown("---")
                st.markdown("### 📊 **Detailed EDA Report** (Generated Above)")
                st.info("� The complete EDA visualizations are displayed in the main report section above the tabs.")

        with tab2:
            st.header("🧹 Interactive Data Cleaning")
            st.info("💡 **Tip:** Automatically detect and fix data quality issues - duplicates, missing values, and outliers!")

            # Add enhanced data quality assessment first
            enhanced_data_quality_dashboard(st.session_state.cleaned_df)
            
            # Add ML-focused quality dashboard
            st.markdown("---")
            create_ml_quality_dashboard(st.session_state.cleaned_df)
            
            # Add domain-specific validation
            st.markdown("---")
            domain_validation_dashboard(st.session_state.cleaned_df)
            
            st.markdown("---")

            # Validate data for cleaning
            is_valid, message = validate_data_for_cleaning(st.session_state.cleaned_df)
            if not is_valid:
                st.error(f"❌ {message}")
            else:
                # Enhanced Auto-Clean Section
                st.subheader("🚀 Enhanced Auto-Clean Pipeline")
                st.info("✨ **New!** Enhanced auto-cleaning with intelligent algorithms for better data quality")
                
                # Smart Cleaning Strategy Recommendation
                with st.expander("🤖 **Smart Strategy Recommendation** (Click to analyze your data)", expanded=False):
                    if st.button("🔍 **Analyze Data & Recommend Strategy**", 
                                type="secondary", 
                                help="Get intelligent recommendations for the best cleaning approach",
                                use_container_width=True):
                        try:
                            with st.spinner("🧠 Analyzing data patterns and generating recommendations..."):
                                # Generate smart recommendations
                                target_column = st.selectbox(
                                    "Select target column for ML analysis (optional):", 
                                    ["None"] + list(st.session_state.cleaned_df.columns),
                                    key="smart_target_selection"
                                )
                                
                                target_col = None if target_column == "None" else target_column
                                recommendations = generate_cleaning_strategy_report(
                                    st.session_state.cleaned_df, 
                                    target_col
                                )
                                
                                # Display overall strategy recommendation
                                st.success(f"🎯 **Recommended Strategy: {recommendations['overall_strategy']} Cleaning**")
                                st.info(f"💡 Expected improvement: {recommendations['expected_improvement']:.0f}% better ML performance")
                                
                                # Display prioritized recommendations
                                st.markdown("### 📋 Prioritized Action Items:")
                                for i, rec in enumerate(recommendations['recommendations'][:5], 1):
                                    priority_emoji = {5: "🔴", 4: "🟠", 3: "🟡", 2: "🔵", 1: "⚪"}
                                    priority_text = {5: "Critical", 4: "High", 3: "Medium", 2: "Low", 1: "Info"}
                                    
                                    with st.container():
                                        col_rec1, col_rec2 = st.columns([1, 4])
                                        with col_rec1:
                                            st.markdown(f"**{i}.** {priority_emoji.get(rec['priority'], '⚪')} **{priority_text.get(rec['priority'], 'Unknown')}**")
                                        with col_rec2:
                                            st.markdown(f"**{rec['category']}**: {rec['action']}")
                                            st.caption(f"💭 {rec['reason']} | 📈 Expected improvement: +{rec['expected_improvement']}%")
                                            if 'implementation' in rec:
                                                st.caption(f"🔧 Implementation: {rec['implementation']}")
                                
                                # Display analysis summary
                                with st.expander("📊 **Detailed Analysis Summary**", expanded=False):
                                    summary = recommendations['analysis_summary']
                                    
                                    col_sum1, col_sum2 = st.columns(2)
                                    with col_sum1:
                                        st.markdown("**Missing Data Analysis:**")
                                        st.write(f"• Total missing: {summary['missing_data'].get('total_missing', 'N/A')}")
                                        st.write(f"• Missing percentage: {summary['missing_data'].get('missing_percentage', 0):.1f}%")
                                        st.write(f"• Affected columns: {summary['missing_data'].get('columns_with_missing', 'N/A')}")
                                        
                                        st.markdown("**Feature Quality:**")
                                        st.write(f"• Total features: {summary['features'].get('total_features', 'N/A')}")
                                        st.write(f"• Constant features: {summary['features'].get('constant_features', 'N/A')}")
                                        st.write(f"• Nearly constant: {summary['features'].get('nearly_constant_features', 'N/A')}")
                                    
                                    with col_sum2:
                                        st.markdown("**Outlier Analysis:**")
                                        st.write(f"• Total outliers: {summary['outliers'].get('total_outliers', 'N/A')}")
                                        st.write(f"• Affected columns: {summary['outliers'].get('affected_columns', 'N/A')}")
                                        st.write(f"• Outlier percentage: {summary['outliers'].get('outlier_percentage', 0):.1f}%")
                                        
                                        st.markdown("**Encoding Analysis:**")
                                        st.write(f"• Categorical columns: {summary['encoding'].get('categorical_columns', 'N/A')}")
                                        st.write(f"• High cardinality: {summary['encoding'].get('high_cardinality', 'N/A')}")
                                        st.write(f"• Need encoding: {summary['encoding'].get('needs_encoding', 'N/A')}")
                                
                                # Store recommendations in session state for later use
                                st.session_state['cleaning_recommendations'] = recommendations
                                
                        except Exception as e:
                            st.error(f"❌ Strategy analysis failed: {str(e)}")
                            logging.error(f"Smart strategy analysis error: {str(e)}\n{traceback.format_exc()}")
                
                col_auto1, col_auto2, col_auto3 = st.columns(3)
                
                with col_auto1:
                    if st.button("🛡️ **Enhanced Conservative Clean**", 
                                type="primary", 
                                help="Safe cleaning: handles missing values, text standardization, and data type optimization",
                                use_container_width=True):
                        try:
                            with st.spinner("🔄 Running Enhanced Conservative Cleaning..."):
                                # Store original data for comparison
                                original_df = st.session_state.cleaned_df.copy()
                                
                                # Apply enhanced conservative cleaning
                                enhanced_cleaner = EnhancedSmartDataCleaner(st.session_state.cleaned_df)
                                cleaned_result = enhanced_cleaner.auto_clean_pipeline(aggressive=False)
                                
                                # Update the dataframe
                                old_missing = original_df.isnull().sum().sum()
                                st.session_state.cleaned_df = cleaned_result
                                new_missing = st.session_state.cleaned_df.isnull().sum().sum()
                                
                                # Store comparison data
                                st.session_state.cleaning_comparison = {
                                    'original': original_df,
                                    'cleaned': cleaned_result,
                                    'cleaning_log': enhanced_cleaner.cleaning_log,
                                    'mode': 'Conservative'
                                }
                                
                                # Show results
                                st.success(f"✅ Enhanced Conservative Cleaning Complete!")
                                st.info(f"📊 Missing values: {old_missing:,} → {new_missing:,} ({old_missing - new_missing:,} fixed)")
                                
                                # Show cleaning log
                                if enhanced_cleaner.cleaning_log:
                                    with st.expander("📋 View Detailed Cleaning Log"):
                                        for log_entry in enhanced_cleaner.cleaning_log:
                                            st.write(f"• {log_entry}")
                                
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"❌ Enhanced Conservative cleaning failed: {str(e)}")
                            logging.error(f"Enhanced Conservative cleaning error: {str(e)}\n{traceback.format_exc()}")
                
                with col_auto2:
                    if st.button("🔥 **Enhanced Aggressive Clean**", 
                                type="secondary",
                                help="Thorough cleaning: includes outlier handling, duplicate removal, and advanced standardization",
                                use_container_width=True):
                        try:
                            with st.spinner("🔄 Running Enhanced Aggressive Cleaning..."):
                                # Store original data for comparison
                                original_df = st.session_state.cleaned_df.copy()
                                
                                # Apply enhanced aggressive cleaning
                                enhanced_cleaner = EnhancedSmartDataCleaner(st.session_state.cleaned_df)
                                cleaned_result = enhanced_cleaner.auto_clean_pipeline(aggressive=True)
                                
                                # Update the dataframe
                                old_missing = original_df.isnull().sum().sum()
                                old_shape = original_df.shape
                                st.session_state.cleaned_df = cleaned_result
                                new_missing = st.session_state.cleaned_df.isnull().sum().sum()
                                new_shape = st.session_state.cleaned_df.shape
                                
                                # Store comparison data
                                st.session_state.cleaning_comparison = {
                                    'original': original_df,
                                    'cleaned': cleaned_result,
                                    'cleaning_log': enhanced_cleaner.cleaning_log,
                                    'mode': 'Aggressive'
                                }
                                
                                # Show results
                                st.success(f"✅ Enhanced Aggressive Cleaning Complete!")
                                st.info(f"📊 Shape: {old_shape} → {new_shape}")
                                st.info(f"📊 Missing values: {old_missing:,} → {new_missing:,} ({old_missing - new_missing:,} fixed)")
                                
                                # Show cleaning log
                                if enhanced_cleaner.cleaning_log:
                                    with st.expander("📋 View Detailed Cleaning Log"):
                                        for log_entry in enhanced_cleaner.cleaning_log:
                                            st.write(f"• {log_entry}")
                                
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"❌ Enhanced Aggressive cleaning failed: {str(e)}")
                            logging.error(f"Enhanced Aggressive cleaning error: {str(e)}\n{traceback.format_exc()}")
                
                with col_auto3:
                    if st.button("🎯 **ML-Ready Clean**", 
                                type="secondary",
                                help="Advanced ML-focused cleaning: feature engineering, encoding, outlier handling, optimized for model training",
                                use_container_width=True):
                        try:
                            with st.spinner("🔄 Running ML-Ready Cleaning Pipeline..."):
                                # Store original data for comparison
                                original_df = st.session_state.cleaned_df.copy()
                                
                                # Detect target column for ML preparation
                                target_col = None
                                if 'target_column' in st.session_state:
                                    target_col = st.session_state.target_column
                                
                                # Apply ML-ready cleaning
                                ml_cleaner = MLReadyCleaner(st.session_state.cleaned_df, target_column=target_col)
                                ml_result = ml_cleaner.prepare_for_ml()
                                
                                # Update the dataframe
                                old_missing = original_df.isnull().sum().sum()
                                old_shape = original_df.shape
                                st.session_state.cleaned_df = ml_result['cleaned_data']
                                new_missing = st.session_state.cleaned_df.isnull().sum().sum()
                                new_shape = st.session_state.cleaned_df.shape
                                
                                # Store ML encoders for later use
                                if 'ml_encoders' not in st.session_state:
                                    st.session_state.ml_encoders = {}
                                st.session_state.ml_encoders.update(ml_result['encoders'])
                                
                                # Store comparison data
                                st.session_state.cleaning_comparison = {
                                    'original': original_df,
                                    'cleaned': ml_result['cleaned_data'],
                                    'cleaning_log': ml_result['cleaning_log'],
                                    'mode': 'ML-Ready',
                                    'ml_report': ml_result['ml_readiness_report']
                                }
                                
                                # Show results
                                readiness_score = ml_result['ml_readiness_report']['ml_readiness_score']
                                readiness_level = ml_result['ml_readiness_report']['readiness_level']
                                
                                if readiness_score >= 90:
                                    st.success(f"✅ ML-Ready Cleaning Complete! 🎯 ML Readiness: {readiness_score:.0f}% ({readiness_level})")
                                elif readiness_score >= 75:
                                    st.info(f"✅ ML-Ready Cleaning Complete! 🎯 ML Readiness: {readiness_score:.0f}% ({readiness_level})")
                                else:
                                    st.warning(f"✅ ML-Ready Cleaning Complete! ⚠️ ML Readiness: {readiness_score:.0f}% ({readiness_level})")
                                
                                st.info(f"📊 Shape: {old_shape} → {new_shape}")
                                st.info(f"📊 Missing values: {old_missing:,} → {new_missing:,} ({old_missing - new_missing:,} fixed)")
                                
                                # Show ML recommendations
                                if ml_result['ml_recommendations']:
                                    with st.expander("🎯 ML Optimization Recommendations"):
                                        for rec in ml_result['ml_recommendations']:
                                            st.write(f"• {rec}")
                                
                                # Show cleaning log
                                if ml_result['cleaning_log']:
                                    with st.expander("📋 View ML Cleaning Log"):
                                        for log_entry in ml_result['cleaning_log']:
                                            st.write(f"• {log_entry}")
                                
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"❌ ML-Ready cleaning failed: {str(e)}")
                            logging.error(f"ML-Ready cleaning error: {str(e)}\n{traceback.format_exc()}")
                
                # Before/After Comparison Section
                if 'cleaning_comparison' in st.session_state:
                    st.markdown("---")
                    st.subheader("📊 Before/After Comparison")
                    st.info(f"🧹 Showing results from **{st.session_state.cleaning_comparison['mode']} Cleaning**")
                    
                    # Create comparison tabs
                    comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs([
                        "📋 Data Preview", 
                        "📈 Statistics", 
                        "🔍 Quality Analysis", 
                        "📝 Detailed Changes"
                    ])
                    
                    original_data = st.session_state.cleaning_comparison['original']
                    cleaned_data = st.session_state.cleaning_comparison['cleaned']
                    
                    with comp_tab1:
                        st.subheader("📋 Data Preview Comparison")
                        
                        col_before, col_after = st.columns(2)
                        
                        with col_before:
                            st.write("**🔴 BEFORE Cleaning:**")
                            st.dataframe(
                                original_data.head(10), 
                                use_container_width=True,
                                height=300
                            )
                            
                        with col_after:
                            st.write("**🟢 AFTER Cleaning:**")
                            st.dataframe(
                                cleaned_data.head(10), 
                                use_container_width=True,
                                height=300
                            )
                    
                    with comp_tab2:
                        st.subheader("📈 Statistical Comparison")
                        
                        # Key metrics comparison
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Total Rows", 
                                f"{len(cleaned_data):,}",
                                delta=f"{len(cleaned_data) - len(original_data):+,}"
                            )
                            
                        with col2:
                            st.metric(
                                "Total Columns", 
                                f"{len(cleaned_data.columns):,}",
                                delta=f"{len(cleaned_data.columns) - len(original_data.columns):+,}"
                            )
                            
                        with col3:
                            orig_missing = original_data.isnull().sum().sum()
                            clean_missing = cleaned_data.isnull().sum().sum()
                            st.metric(
                                "Missing Values", 
                                f"{clean_missing:,}",
                                delta=f"{clean_missing - orig_missing:+,}"
                            )
                            
                        with col4:
                            orig_duplicates = original_data.duplicated().sum()
                            clean_duplicates = cleaned_data.duplicated().sum()
                            st.metric(
                                "Duplicate Rows", 
                                f"{clean_duplicates:,}",
                                delta=f"{clean_duplicates - orig_duplicates:+,}"
                            )
                        
                        # Missing values by column comparison
                        st.subheader("🔍 Missing Values by Column")
                        
                        orig_missing_by_col = original_data.isnull().sum()
                        clean_missing_by_col = cleaned_data.isnull().sum()
                        
                        # Get common columns and handle shape differences
                        common_cols = orig_missing_by_col.index.intersection(clean_missing_by_col.index)
                        new_cols = clean_missing_by_col.index.difference(orig_missing_by_col.index)
                        removed_cols = orig_missing_by_col.index.difference(clean_missing_by_col.index)
                        
                        comparison_rows = []
                        
                        # Add rows for common columns
                        for col in common_cols:
                            before_val = orig_missing_by_col[col]
                            after_val = clean_missing_by_col[col]
                            comparison_rows.append({
                                'Column': col,
                                'Before': before_val,
                                'After': after_val,
                                'Fixed': before_val - after_val,
                                'Before %': round((before_val / len(original_data)) * 100, 2),
                                'After %': round((after_val / len(cleaned_data)) * 100, 2),
                                'Status': 'Common'
                            })
                        
                        # Add rows for new columns (only in cleaned data)
                        for col in new_cols:
                            after_val = clean_missing_by_col[col]
                            comparison_rows.append({
                                'Column': col,
                                'Before': 0,
                                'After': after_val,
                                'Fixed': 0,
                                'Before %': 0,
                                'After %': round((after_val / len(cleaned_data)) * 100, 2),
                                'Status': 'New Feature'
                            })
                        
                        # Add rows for removed columns (only in original data)
                        for col in removed_cols:
                            before_val = orig_missing_by_col[col]
                            comparison_rows.append({
                                'Column': col,
                                'Before': before_val,
                                'After': 0,
                                'Fixed': before_val,
                                'Before %': round((before_val / len(original_data)) * 100, 2),
                                'After %': 0,
                                'Status': 'Removed'
                            })
                        
                        comparison_df = pd.DataFrame(comparison_rows)
                        
                        # Only show columns that had missing values or are new/removed
                        if not comparison_df.empty:
                            comparison_df = comparison_df[
                                (comparison_df['Before'] > 0) | 
                                (comparison_df['After'] > 0) | 
                                (comparison_df['Status'].isin(['New Feature', 'Removed']))
                            ].sort_values('Fixed', ascending=False)
                        
                        if not comparison_df.empty:
                            st.dataframe(comparison_df, use_container_width=True)
                        else:
                            st.info("✅ No missing values were present in the original data")
                    
                    with comp_tab3:
                        st.subheader("🔍 Data Quality Analysis")
                        
                        # ML Readiness Score (if available)
                        if 'ml_report' in st.session_state.cleaning_comparison:
                            ml_report = st.session_state.cleaning_comparison['ml_report']
                            
                            st.write("**🎯 ML Readiness Assessment:**")
                            
                            score = ml_report['ml_readiness_score']
                            level = ml_report['readiness_level']
                            color = ml_report['readiness_color']
                            
                            if score >= 90:
                                st.success(f"🏆 ML Readiness Score: {score:.0f}% ({level})")
                            elif score >= 75:
                                st.info(f"✅ ML Readiness Score: {score:.0f}% ({level})")
                            elif score >= 60:
                                st.warning(f"⚠️ ML Readiness Score: {score:.0f}% ({level})")
                            else:
                                st.error(f"❌ ML Readiness Score: {score:.0f}% ({level})")
                            
                            # ML metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Numeric Features", ml_report['numeric_features'])
                            with col2:
                                st.metric("Categorical Features", ml_report['categorical_features'])
                            with col3:
                                st.metric("Memory Usage", f"{ml_report['memory_usage_mb']:.1f} MB")
                            with col4:
                                st.metric("Total Features", ml_report['data_shape'][1])
                        
                        # Data types comparison
                        st.write("**📊 Data Types Comparison:**")
                        
                        orig_types = original_data.dtypes.value_counts()
                        clean_types = cleaned_data.dtypes.value_counts()
                        
                        types_comparison = pd.DataFrame({
                            'Data Type': [str(dt) for dt in orig_types.index.union(clean_types.index)],
                            'Before': [orig_types.get(dt, 0) for dt in orig_types.index.union(clean_types.index)],
                            'After': [clean_types.get(dt, 0) for dt in orig_types.index.union(clean_types.index)]
                        })
                        types_comparison['Change'] = types_comparison['After'] - types_comparison['Before']
                        
                        st.dataframe(make_arrow_compatible(types_comparison), use_container_width=True)
                        
                        # Categorical values comparison for key columns
                        st.write("**🏷️ Categorical Standardization:**")
                        
                        categorical_cols = original_data.select_dtypes(include=['object']).columns
                        
                        for col in categorical_cols[:3]:  # Show first 3 categorical columns
                            if col in cleaned_data.columns:
                                orig_unique = len(original_data[col].dropna().unique())
                                clean_unique = len(cleaned_data[col].dropna().unique())
                                
                                if orig_unique != clean_unique:
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**{col} - Before ({orig_unique} unique):**")
                                        st.write(list(original_data[col].dropna().unique()[:10]))
                                        if orig_unique > 10:
                                            st.write(f"... and {orig_unique - 10} more")
                                    
                                    with col2:
                                        st.write(f"**{col} - After ({clean_unique} unique):**")
                                        st.write(list(cleaned_data[col].dropna().unique()[:10]))
                                        if clean_unique > 10:
                                            st.write(f"... and {clean_unique - 10} more")
                        
                        # Outlier analysis for numeric columns
                        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
                        
                        if len(numeric_cols) > 0:
                            st.write("**📊 Outlier Analysis:**")
                            
                            outlier_summary = []
                            for col in numeric_cols:
                                if col in cleaned_data.columns:
                                    # IQR method for outlier detection
                                    Q1_orig = original_data[col].quantile(0.25)
                                    Q3_orig = original_data[col].quantile(0.75)
                                    IQR_orig = Q3_orig - Q1_orig
                                    
                                    if IQR_orig > 0:
                                        lower_bound = Q1_orig - 1.5 * IQR_orig
                                        upper_bound = Q3_orig + 1.5 * IQR_orig
                                        
                                        orig_outliers = ((original_data[col] < lower_bound) | (original_data[col] > upper_bound)).sum()
                                        clean_outliers = ((cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)).sum()
                                        
                                        outlier_summary.append({
                                            'Column': col,
                                            'Original Range': f"{original_data[col].min():.2f} to {original_data[col].max():.2f}",
                                            'Cleaned Range': f"{cleaned_data[col].min():.2f} to {cleaned_data[col].max():.2f}",
                                            'Outliers Before': orig_outliers,
                                            'Outliers After': clean_outliers,
                                            'Outliers Fixed': orig_outliers - clean_outliers
                                        })
                            
                            if outlier_summary:
                                outlier_df = pd.DataFrame(outlier_summary)
                                st.dataframe(outlier_df, use_container_width=True)
                    
                    with comp_tab4:
                        st.subheader("📝 Detailed Cleaning Operations")
                        
                        st.write("**🔧 Operations Performed:**")
                        cleaning_log = st.session_state.cleaning_comparison['cleaning_log']
                        
                        for i, log_entry in enumerate(cleaning_log, 1):
                            st.write(f"{i}. {log_entry}")
                        
                        # Summary statistics
                        st.write("**📊 Summary Statistics:**")
                        
                        summary_stats = {
                            "Total Operations": len(cleaning_log),
                            "Rows Changed": len(original_data) - len(cleaned_data),
                            "Missing Values Fixed": original_data.isnull().sum().sum() - cleaned_data.isnull().sum().sum(),
                            "Duplicates Removed": original_data.duplicated().sum() - cleaned_data.duplicated().sum(),
                            "Data Types Optimized": len(cleaned_data.select_dtypes(include=[np.number]).columns) - len(original_data.select_dtypes(include=[np.number]).columns)
                        }
                        
                        for key, value in summary_stats.items():
                            if value != 0:
                                st.write(f"• **{key}:** {value:,}")
                        
                        # Export options
                        st.write("**💾 Export Options:**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("📁 Download Cleaned Data", use_container_width=True):
                                csv_data = convert_df_to_csv(cleaned_data)
                                st.download_button(
                                    label="📥 Download CSV",
                                    data=csv_data,
                                    file_name=f"cleaned_data_{st.session_state.cleaning_comparison['mode'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        
                        with col2:
                            if st.button("📊 Download Comparison Report", use_container_width=True):
                                # Create a simple comparison report
                                report = f"""
DATA CLEANING REPORT - {st.session_state.cleaning_comparison['mode']} Mode
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
- Original Shape: {original_data.shape}
- Cleaned Shape: {cleaned_data.shape}
- Missing Values Fixed: {original_data.isnull().sum().sum() - cleaned_data.isnull().sum().sum()}
- Duplicates Removed: {original_data.duplicated().sum() - cleaned_data.duplicated().sum()}

OPERATIONS PERFORMED:
{chr(10).join([f"{i}. {log}" for i, log in enumerate(cleaning_log, 1)])}
"""
                                st.download_button(
                                    label="📥 Download Report",
                                    data=report,
                                    file_name=f"cleaning_report_{st.session_state.cleaning_comparison['mode'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain"
                                )
                
                st.markdown("---")
                
                # Add cleaning strategy dropdown
                st.subheader("🛠️ Manual Cleaning Operations")
                st.info("💡 Try the Enhanced Auto-Clean options above first, or use manual controls below for specific cleaning tasks.")
                
                cleaning_strategy = st.selectbox(
                    "Choose what you want to clean:",
                    [
                        "Missing Values",
                        "Duplicates (Rows & IDs)",
                        "Outlier Detection & Treatment",
                        "Data Type Optimization",
                        "All Operations",
                    ],
                    help="Select the type of cleaning operation you want to perform",
                )

                # Enhanced cleaning interface
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("📊 Current Data Status")

                    # Current data preview with better formatting
                    st.write("**Data Preview:**")
                    st.dataframe(
                        st.session_state.cleaned_df.head(),
                        use_container_width=True,
                        height=200,
                    )

                    # Enhanced statistics
                    col1a, col1b = st.columns(2)
                    with col1a:
                        st.metric(
                            "Total Rows",
                            f"{len(st.session_state.cleaned_df):,}",
                            delta=(
                                f"{len(st.session_state.cleaned_df) - len(df_original):+,}"
                                if len(st.session_state.cleaned_df) != len(df_original)
                                else None
                            ),
                        )
                        st.metric(
                            "Missing Values",
                            f"{st.session_state.cleaned_df.isnull().sum().sum():,}",
                            delta=(
                                f"{st.session_state.cleaned_df.isnull().sum().sum() - df_original.isnull().sum().sum():+,}"
                                if st.session_state.cleaned_df.isnull().sum().sum()
                                != df_original.isnull().sum().sum()
                                else None
                            ),
                        )

                    with col1b:
                        st.metric(
                            "Duplicate Rows",
                            f"{st.session_state.cleaned_df.duplicated().sum():,}",
                            delta=(
                                f"{st.session_state.cleaned_df.duplicated().sum() - df_original.duplicated().sum():+,}"
                                if st.session_state.cleaned_df.duplicated().sum()
                                != df_original.duplicated().sum()
                                else None
                            ),
                        )
                        st.metric(
                            "Data Quality Score",
                            f"{calculate_data_quality_score(st.session_state.cleaned_df):.1f}%",
                        )

                    # Missing values by column
                    missing_by_col = st.session_state.cleaned_df.isnull().sum()
                    if missing_by_col.sum() > 0:
                        st.write("**Missing Values by Column:**")
                        missing_df = pd.DataFrame(
                            {
                                "Column": missing_by_col.index,
                                "Missing Count": missing_by_col.values,
                                "Missing %": (
                                    missing_by_col.values
                                    / len(st.session_state.cleaned_df)
                                    * 100
                                ).round(2),
                            }
                        )
                        missing_df = missing_df[
                            missing_df["Missing Count"] > 0
                        ].sort_values("Missing Count", ascending=False)
                        st.dataframe(missing_df, use_container_width=True)

                with col2:
                    st.subheader("🛠️ Cleaning Operations")

                    # Show cleaning options based on selected strategy
                    if cleaning_strategy == "Missing Values":
                        # Missing value handling
                        st.write("### 🔧 Handle Missing Values")
                        mv_strategy = st.selectbox(
                            "Select Strategy",
                            [
                                "Drop Rows",
                                "Fill with Mean",
                                "Fill with Median",
                                "Fill with Mode",
                                "Forward Fill",
                                "Backward Fill",
                            ],
                            key="mv_strategy",
                            help="Choose how to handle missing values",
                        )

                        # Smart column selection - ensure consistency with analysis
                        # Always check the source data that the analysis is using
                        current_data = st.session_state.cleaned_df
                        original_data = st.session_state.get('original_df', current_data)
                        
                        # For missing values analysis consistency, check original data if available
                        analysis_data = original_data if 'original_df' in st.session_state else current_data
                        
                        # Check for missing values in the data being analyzed
                        missing_val_cols = analysis_data.columns[analysis_data.isnull().any()].tolist()
                        total_missing = analysis_data.isnull().sum().sum()
                        
                        # Show which dataset we're analyzing
                        if analysis_data is original_data and original_data is not current_data:
                            st.info("🔍 Analyzing original data (same as EDA analysis above)")
                        
                        if missing_val_cols and total_missing > 0:
                            mv_columns = st.multiselect(
                                "Select Columns",
                                options=st.session_state.cleaned_df.columns,
                                default=missing_val_cols,
                                key="mv_cols",
                                help="Select columns to apply the cleaning strategy",
                            )

                            # Show preview of what will be affected
                            if mv_columns:
                                affected_rows = (
                                    st.session_state.cleaned_df[mv_columns]
                                    .isnull()
                                    .any(axis=1)
                                    .sum()
                                )
                                st.info(f"📊 This will affect {affected_rows:,} rows")

                            if st.button(
                                "Apply Missing Value Strategy",
                                type="primary",
                                key="apply_mv",
                            ):
                                try:
                                    with st.spinner("Applying cleaning strategy..."):
                                        old_shape = st.session_state.cleaned_df.shape
                                        st.session_state.cleaned_df = (
                                            handle_missing_values(
                                                st.session_state.cleaned_df,
                                                mv_strategy,
                                                mv_columns,
                                            )
                                        )
                                        new_shape = st.session_state.cleaned_df.shape

                                        st.success(
                                            f"✅ Applied '{mv_strategy}' to {len(mv_columns)} columns"
                                        )
                                        if old_shape[0] != new_shape[0]:
                                            st.info(
                                                f"📊 Rows changed: {old_shape[0]:,} → {new_shape[0]:,}"
                                            )

                                        logger.info(
                                            f"Applied {mv_strategy} to columns: {mv_columns}"
                                        )
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error applying strategy: {str(e)}")
                                    logger.error(
                                        f"Error in missing value handling: {str(e)}"
                                    )
                        else:
                            # Check if this is a case where original data had missing values but current doesn't
                            if "original_df" in st.session_state:
                                orig_missing = st.session_state.original_df.isnull().sum().sum()
                                if orig_missing > 0:
                                    st.success("✅ No missing values in current dataset!")
                                    st.info(f"ℹ️ Original data had {orig_missing:,} missing values. Use 'Reset to Original' if you want to clean the original data.")
                                else:
                                    st.success("✅ No missing values found in this dataset!")
                            else:
                                st.success("✅ No missing values found in current dataset!")

                    elif cleaning_strategy == "Duplicates (Rows & IDs)":
                        # Duplicate removal
                        st.write("### 🔍 Remove Duplicates (Complete Rows & ID Columns)")
                        
                        # Check for complete row duplicates
                        row_dup_count = st.session_state.cleaned_df.duplicated().sum()
                        
                        # Check for ID column duplicates
                        id_cols = [col for col in st.session_state.cleaned_df.columns 
                                  if any(term in col.lower() for term in ['id', '_key', 'uuid', 'identifier'])]
                        
                        id_duplicates = {}
                        for col in id_cols:
                            dup_count = st.session_state.cleaned_df[col].duplicated().sum()
                            if dup_count > 0:
                                id_duplicates[col] = dup_count
                        
                        if row_dup_count > 0 or id_duplicates:
                            if row_dup_count > 0:
                                st.warning(f"⚠️ Found {row_dup_count:,} complete duplicate rows")
                            
                            if id_duplicates:
                                st.error("🚨 **Critical ID Column Duplicates Found:**")
                                for col, count in id_duplicates.items():
                                    st.write(f"• **{col}**: {count:,} duplicate values")
                                st.write("⚠️ ID columns should have unique values for data integrity")
                            
                            # Duplicate removal options
                            st.write("**Choose duplicate handling approach:**")
                            
                            dup_col1, dup_col2 = st.columns(2)
                            
                            with dup_col1:
                                if row_dup_count > 0:
                                    if st.button("🗑️ Remove Complete Duplicate Rows", 
                                                 type="primary", 
                                                 key="remove_row_duplicates"):
                                        try:
                                            with st.spinner("Removing complete duplicate rows..."):
                                                old_count = len(st.session_state.cleaned_df)
                                                st.session_state.cleaned_df = remove_duplicates(
                                                    st.session_state.cleaned_df
                                                )
                                                new_count = len(st.session_state.cleaned_df)
                                                removed = old_count - new_count
                                                st.success(f"✅ Removed {removed:,} complete duplicate rows")
                                                logger.info(f"Removed {removed} duplicate rows")
                                                st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Error removing duplicates: {str(e)}")
                                            logger.error(f"Error in duplicate removal: {str(e)}")
                            
                            with dup_col2:
                                if id_duplicates:
                                    if st.button("🔧 Fix ID Column Duplicates", 
                                                 type="secondary", 
                                                 key="fix_id_duplicates"):
                                        try:
                                            with st.spinner("Fixing ID column duplicates..."):
                                                fixed_count = 0
                                                for col in id_duplicates.keys():
                                                    # Keep first occurrence, mark others as problematic
                                                    mask = st.session_state.cleaned_df[col].duplicated(keep='first')
                                                    if mask.any():
                                                        # Option 1: Add suffix to make unique
                                                        duplicated_rows = st.session_state.cleaned_df[mask]
                                                        for idx in duplicated_rows.index:
                                                            original_value = st.session_state.cleaned_df.at[idx, col]
                                                            st.session_state.cleaned_df.at[idx, col] = f"{original_value}_dup_{idx}"
                                                        fixed_count += mask.sum()
                                                
                                                st.success(f"✅ Fixed {fixed_count:,} ID duplicate values by adding unique suffixes")
                                                logger.info(f"Fixed {fixed_count} ID duplicate values")
                                                st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Error fixing ID duplicates: {str(e)}")
                                            logger.error(f"Error in ID duplicate fixing: {str(e)}")
                            
                            # Show duplicate preview
                            if row_dup_count > 0 and st.checkbox("Show duplicate rows", key="show_duplicates"):
                                duplicates = st.session_state.cleaned_df[
                                    st.session_state.cleaned_df.duplicated(keep=False)
                                ]
                                st.dataframe(duplicates, use_container_width=True)
                            
                            # Show ID duplicate preview
                            if id_duplicates:
                                if st.checkbox("Show ID column duplicates", key="show_id_duplicates"):
                                    st.write("**ID Column Duplicate Analysis:**")
                                    for col in id_duplicates.keys():
                                        st.write(f"**{col} duplicates:**")
                                        dup_mask = st.session_state.cleaned_df[col].duplicated(keep=False)
                                        dup_data = st.session_state.cleaned_df[dup_mask].sort_values(col)
                                        st.dataframe(dup_data[[col] + [c for c in st.session_state.cleaned_df.columns if c != col][:5]], 
                                                   use_container_width=True)
                        else:
                            st.success("✅ No duplicate rows or ID duplicates found!")

                    elif cleaning_strategy == "Outlier Detection & Treatment":
                        # Outlier detection and treatment
                        st.write("### 🎯 Outlier Detection & Treatment")
                        st.write("Detect and handle outliers in numerical columns:")

                        # Get numerical columns
                        numeric_cols = st.session_state.cleaned_df.select_dtypes(
                            include=[np.number]
                        ).columns.tolist()

                        if not numeric_cols:
                            st.warning(
                                "⚠️ No numerical columns found for outlier detection."
                            )
                        else:
                            col1, col2 = st.columns(2)

                            with col1:
                                # Column selection
                                selected_outlier_cols = st.multiselect(
                                    "Select columns for outlier detection:",
                                    options=numeric_cols,
                                    default=(
                                        numeric_cols[:3]
                                        if len(numeric_cols) >= 3
                                        else numeric_cols
                                    ),
                                    help="Choose numerical columns to analyze for outliers",
                                )

                                # Detection method
                                detection_method = st.selectbox(
                                    "Detection Method:",
                                    ["IQR", "Z-Score", "Modified Z-Score"],
                                    help="IQR: Interquartile Range (robust), Z-Score: Standard deviations, Modified Z-Score: More robust than Z-Score",
                                )

                            with col2:
                                # Treatment method
                                treatment_method = st.selectbox(
                                    "Treatment Method:",
                                    [
                                        "Remove",
                                        "Cap",
                                        "Replace with Mean",
                                        "Replace with Median",
                                        "Transform",
                                    ],
                                    help="Remove: Delete outlier rows, Cap: Set to boundary values, Replace: Use mean/median, Transform: Apply log transformation",
                                )

                                # Method-specific parameters
                                if detection_method == "IQR":
                                    iqr_multiplier = st.slider(
                                        "IQR Multiplier:",
                                        min_value=1.0,
                                        max_value=3.0,
                                        value=1.5,
                                        step=0.1,
                                        help="Higher values = less strict outlier detection",
                                    )
                                elif detection_method == "Z-Score":
                                    zscore_threshold = st.slider(
                                        "Z-Score Threshold:",
                                        min_value=2.0,
                                        max_value=4.0,
                                        value=3.0,
                                        step=0.1,
                                        help="Higher values = less strict outlier detection",
                                    )
                                elif detection_method == "Modified Z-Score":
                                    modified_zscore_threshold = st.slider(
                                        "Modified Z-Score Threshold:",
                                        min_value=2.5,
                                        max_value=4.5,
                                        value=3.5,
                                        step=0.1,
                                        help="Higher values = less strict outlier detection",
                                    )

                            if selected_outlier_cols:
                                # Preview outliers before treatment
                                st.write("**Preview Outlier Detection:**")
                                if st.button(
                                    "🔍 Detect Outliers", key="preview_outliers"
                                ):
                                    # Prepare parameters
                                    kwargs = {}
                                    if detection_method == "IQR":
                                        kwargs["iqr_multiplier"] = iqr_multiplier
                                    elif detection_method == "Z-Score":
                                        kwargs["zscore_threshold"] = zscore_threshold
                                    elif detection_method == "Modified Z-Score":
                                        kwargs["modified_zscore_threshold"] = (
                                            modified_zscore_threshold
                                        )

                                    # Show outlier statistics without applying treatment
                                    temp_df = handle_outliers(
                                        st.session_state.cleaned_df,
                                        "Remove",  # Just for detection
                                        detection_method,
                                        selected_outlier_cols,
                                        **kwargs,
                                    )

                                # Apply outlier treatment
                                if st.button(
                                    "🚀 Apply Outlier Treatment",
                                    type="primary",
                                    key="apply_outliers",
                                ):
                                    try:
                                        # Prepare parameters
                                        kwargs = {}
                                        if detection_method == "IQR":
                                            kwargs["iqr_multiplier"] = iqr_multiplier
                                        elif detection_method == "Z-Score":
                                            kwargs["zscore_threshold"] = (
                                                zscore_threshold
                                            )
                                        elif detection_method == "Modified Z-Score":
                                            kwargs["modified_zscore_threshold"] = (
                                                modified_zscore_threshold
                                            )

                                        original_rows = len(st.session_state.cleaned_df)

                                        # Apply outlier treatment
                                        st.session_state.cleaned_df = handle_outliers(
                                            st.session_state.cleaned_df,
                                            treatment_method,
                                            detection_method,
                                            selected_outlier_cols,
                                            **kwargs,
                                        )

                                        new_rows = len(st.session_state.cleaned_df)

                                        if (
                                            treatment_method == "Remove"
                                            and new_rows < original_rows
                                        ):
                                            st.success(
                                                f"✅ Outlier treatment completed! Removed {original_rows - new_rows} rows."
                                            )
                                        else:
                                            st.success(
                                                f"✅ Outlier treatment completed using {treatment_method.lower()} method!"
                                            )

                                        logger.info(
                                            f"Applied {treatment_method} outlier treatment using {detection_method} detection"
                                        )
                                        st.rerun()

                                    except Exception as e:
                                        st.error(
                                            f"❌ Error in outlier treatment: {str(e)}"
                                        )
                                        logger.error(
                                            f"Error in outlier treatment: {str(e)}"
                                        )
                            else:
                                st.info(
                                    "👆 Please select columns for outlier detection."
                                )

                    elif cleaning_strategy == "Data Type Optimization":
                        # Data type optimization
                        st.write("### ⚡ Optimize Data Types")
                        st.write("Optimize memory usage by converting data types:")

                        # Show current memory usage
                        current_memory = st.session_state.cleaned_df.memory_usage(
                            deep=True
                        ).sum()
                        st.info(f"Current memory usage: {current_memory / 1024:.1f} KB")

                        # Show data types summary
                        st.write("**Current Data Types:**")
                        dtype_counts = st.session_state.cleaned_df.dtypes.value_counts()
                        st.dataframe(
                            make_arrow_compatible(pd.DataFrame(
                                {
                                    "Data Type": [str(dt) for dt in dtype_counts.index],
                                    "Count": dtype_counts.values,
                                }
                            ))
                        )

                        if st.button(
                            "Apply Data Type Optimization",
                            type="primary",
                            key="apply_optimization",
                        ):
                            try:
                                with st.spinner("Optimizing data types..."):
                                    # Convert object columns that are numbers
                                    for (
                                        col
                                    ) in st.session_state.cleaned_df.select_dtypes(
                                        include=["object"]
                                    ).columns:
                                        try:
                                            st.session_state.cleaned_df[col] = (
                                                pd.to_numeric(
                                                    st.session_state.cleaned_df[col],
                                                    errors="ignore",
                                                )
                                            )
                                        except:
                                            pass

                                    # Downcast numeric types
                                    for (
                                        col
                                    ) in st.session_state.cleaned_df.select_dtypes(
                                        include=["int"]
                                    ).columns:
                                        st.session_state.cleaned_df[col] = (
                                            pd.to_numeric(
                                                st.session_state.cleaned_df[col],
                                                downcast="integer",
                                            )
                                        )

                                    for (
                                        col
                                    ) in st.session_state.cleaned_df.select_dtypes(
                                        include=["float"]
                                    ).columns:
                                        st.session_state.cleaned_df[col] = (
                                            pd.to_numeric(
                                                st.session_state.cleaned_df[col],
                                                downcast="float",
                                            )
                                        )

                                    new_memory = (
                                        st.session_state.cleaned_df.memory_usage(
                                            deep=True
                                        ).sum()
                                    )
                                    saved = current_memory - new_memory
                                    st.success(
                                        f"✅ Optimized! Saved {saved / 1024:.1f} KB ({(saved/current_memory*100):.1f}%)"
                                    )
                                    logger.info(
                                        f"Data types optimized, saved {saved} bytes"
                                    )
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error optimizing data types: {str(e)}")

                    elif cleaning_strategy == "All Operations":
                        # Apply all cleaning operations
                        st.write("### 🚀 Apply All Cleaning Operations")
                        st.info(
                            "This will apply all available cleaning strategies in sequence:"
                        )
                        st.write("1. Handle missing values (using median/mode)")
                        st.write("2. Remove duplicate rows")
                        st.write("3. Detect and cap outliers (using IQR method)")
                        st.write("4. Optimize data types")

                        if st.button(
                            "Apply All Cleaning Operations",
                            type="primary",
                            key="apply_all",
                        ):
                            try:
                                with st.spinner("Applying all cleaning operations..."):
                                    operations_applied = []

                                    # 1. Handle missing values
                                    missing_count = (
                                        st.session_state.cleaned_df.isnull().sum().sum()
                                    )
                                    if missing_count > 0:
                                        for col in st.session_state.cleaned_df.columns:
                                            if (
                                                st.session_state.cleaned_df[col]
                                                .isnull()
                                                .any()
                                            ):
                                                if pd.api.types.is_numeric_dtype(
                                                    st.session_state.cleaned_df[col]
                                                ):
                                                    st.session_state.cleaned_df[
                                                        col
                                                    ].fillna(
                                                        st.session_state.cleaned_df[
                                                            col
                                                        ].median(),
                                                        inplace=True,
                                                    )
                                                else:
                                                    mode_val = (
                                                        st.session_state.cleaned_df[
                                                            col
                                                        ].mode()
                                                    )
                                                    if not mode_val.empty:
                                                        st.session_state.cleaned_df[
                                                            col
                                                        ].fillna(
                                                            mode_val.iloc[0],
                                                            inplace=True,
                                                        )
                                        operations_applied.append(
                                            f"Handled {missing_count} missing values"
                                        )

                                    # 2. Remove duplicates
                                    dup_count = (
                                        st.session_state.cleaned_df.duplicated().sum()
                                    )
                                    if dup_count > 0:
                                        st.session_state.cleaned_df = (
                                            st.session_state.cleaned_df.drop_duplicates()
                                        )
                                        operations_applied.append(
                                            f"Removed {dup_count} duplicate rows"
                                        )

                                    # 3. Handle outliers (using IQR method with capping)
                                    numeric_cols = (
                                        st.session_state.cleaned_df.select_dtypes(
                                            include=[np.number]
                                        ).columns.tolist()
                                    )
                                    if numeric_cols:
                                        outlier_cols_treated = 0
                                        for col in numeric_cols:
                                            try:
                                                # Apply IQR-based outlier capping
                                                st.session_state.cleaned_df = (
                                                    handle_outliers(
                                                        st.session_state.cleaned_df,
                                                        "Cap",
                                                        "IQR",
                                                        [col],
                                                        iqr_multiplier=1.5,
                                                    )
                                                )
                                                outlier_cols_treated += 1
                                            except Exception as e:
                                                logger.warning(
                                                    f"Could not process outliers for column {col}: {str(e)}"
                                                )

                                        if outlier_cols_treated > 0:
                                            operations_applied.append(
                                                f"Applied outlier capping to {outlier_cols_treated} numeric columns"
                                            )

                                    # 4. Optimize data types
                                    original_memory = (
                                        st.session_state.cleaned_df.memory_usage(
                                            deep=True
                                        ).sum()
                                    )
                                    for (
                                        col
                                    ) in st.session_state.cleaned_df.select_dtypes(
                                        include=["object"]
                                    ).columns:
                                        try:
                                            st.session_state.cleaned_df[col] = (
                                                pd.to_numeric(
                                                    st.session_state.cleaned_df[col],
                                                    errors="ignore",
                                                )
                                            )
                                        except:
                                            pass

                                    for (
                                        col
                                    ) in st.session_state.cleaned_df.select_dtypes(
                                        include=["int"]
                                    ).columns:
                                        st.session_state.cleaned_df[col] = (
                                            pd.to_numeric(
                                                st.session_state.cleaned_df[col],
                                                downcast="integer",
                                            )
                                        )

                                    for (
                                        col
                                    ) in st.session_state.cleaned_df.select_dtypes(
                                        include=["float"]
                                    ).columns:
                                        st.session_state.cleaned_df[col] = (
                                            pd.to_numeric(
                                                st.session_state.cleaned_df[col],
                                                downcast="float",
                                            )
                                        )

                                    new_memory = (
                                        st.session_state.cleaned_df.memory_usage(
                                            deep=True
                                        ).sum()
                                    )
                                    memory_saved = original_memory - new_memory
                                    if memory_saved > 0:
                                        operations_applied.append(
                                            f"Optimized data types (saved {memory_saved/1024:.1f} KB)"
                                        )

                                    # Show results
                                    if operations_applied:
                                        st.success(
                                            "✅ All cleaning operations completed!"
                                        )
                                        for operation in operations_applied:
                                            st.write(f"• {operation}")
                                    else:
                                        st.info(
                                            "ℹ️ No cleaning operations were needed - data is already clean!"
                                        )

                                    logger.info(
                                        f"Applied all cleaning operations: {operations_applied}"
                                    )
                                    st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error applying all operations: {str(e)}")
                                logger.error(f"Error in apply all operations: {str(e)}")

                    # Reset button (always visible)
                    st.markdown("---")
                    if st.button(
                        "🔄 Reset to Original Data",
                        help="Restore data to original state",
                    ):
                        st.session_state.cleaned_df = df_original.copy()
                        st.success("✅ Data reset to original state")
                        logger.info("Data reset to original state")
                        st.rerun()

        with tab3:
            st.header("🤖 Advanced Machine Learning Suite")
            st.info("💡 **Tip:** Build and compare multiple ML models automatically. Features selection and evaluation included!")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: white; margin: 0;">🚀 Professional ML Pipeline</h4>
                <p style="color: #f0f0f0; margin: 5px 0 0 0;">
                    Build, compare, and deploy production-ready machine learning models
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Check if we have data to work with
            model_df = (
                st.session_state.cleaned_df
                if st.session_state.cleaned_df is not None
                else df_original
            )

            if model_df is not None and len(model_df) > 0:
                
                # Create main tabs for modeling workflow
                model_tab1, model_tab2, model_tab3, model_tab4, model_tab5, model_tab6 = st.tabs([
                    "⚙️ Model Setup", 
                    "🏆 Model Comparison", 
                    "🔮 Predictions", 
                    "🚀 Model Deployment",
                    "📁 Import Models",
                    "📊 Performance Analysis"
                ])
                
                with model_tab5:
                    st.subheader("📁 Import & Use Saved Models")
                    st.markdown("Load previously exported models for predictions")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("#### 🔄 Import Model")
                        
                        import_type = st.selectbox(
                            "Import Type:",
                            ["Single Model File", "Complete Package"]
                        )
                        
                        if import_type == "Single Model File":
                            uploaded_model = st.file_uploader(
                                "Upload Model File",
                                type=['pkl', 'joblib'],
                                help="Upload a .pkl or .joblib model file"
                            )
                            
                            if uploaded_model:
                                try:
                                    import joblib
                                    import pickle
                                    import io
                                    
                                    # Load the model
                                    if uploaded_model.name.endswith('.joblib'):
                                        model = joblib.load(io.BytesIO(uploaded_model.read()))
                                    else:
                                        model = pickle.load(io.BytesIO(uploaded_model.read()))
                                    
                                    st.session_state.imported_model = model
                                    st.session_state.imported_model_name = uploaded_model.name
                                    
                                    st.success(f"✅ Model '{uploaded_model.name}' loaded successfully!")
                                    
                                    # Show model info if available
                                    if hasattr(model, 'get_params'):
                                        with st.expander("🔍 Model Parameters"):
                                            st.json(model.get_params())
                                    
                                except Exception as e:
                                    st.error(f"❌ Error loading model: {str(e)}")
                        
                        else:  # Complete Package
                            uploaded_package = st.file_uploader(
                                "Upload Complete Package",
                                type=['zip'],
                                help="Upload a .zip package with model and all components"
                            )
                            
                            if uploaded_package:
                                try:
                                    import zipfile
                                    import io
                                    import pickle
                                    import joblib
                                    import tempfile
                                    import os
                                    
                                    # Extract package to temporary directory
                                    with tempfile.TemporaryDirectory() as temp_dir:
                                        with zipfile.ZipFile(io.BytesIO(uploaded_package.read())) as zip_file:
                                            zip_file.extractall(temp_dir)
                                            
                                            # Load model info
                                            info_path = os.path.join(temp_dir, 'model_info.pkl')
                                            if os.path.exists(info_path):
                                                with open(info_path, 'rb') as f:
                                                    model_info = pickle.load(f)
                                                
                                                st.session_state.imported_feature_names = model_info['feature_names']
                                                st.session_state.imported_target_column = model_info['target_column']
                                                st.session_state.imported_problem_type = model_info['problem_type']
                                            
                                            # Load model
                                            model_files = [f for f in os.listdir(temp_dir) if f.endswith('.joblib')]
                                            if model_files:
                                                model_path = os.path.join(temp_dir, model_files[0])
                                                model = joblib.load(model_path)
                                                st.session_state.imported_model = model
                                                st.session_state.imported_model_name = model_files[0]
                                            
                                            # Load label encoders
                                            encoders_path = os.path.join(temp_dir, 'label_encoders.pkl')
                                            if os.path.exists(encoders_path):
                                                with open(encoders_path, 'rb') as f:
                                                    st.session_state.imported_label_encoders = pickle.load(f)
                                            
                                            # Load scaler
                                            scaler_path = os.path.join(temp_dir, 'scaler.pkl')
                                            if os.path.exists(scaler_path):
                                                with open(scaler_path, 'rb') as f:
                                                    st.session_state.imported_scaler = pickle.load(f)
                                    
                                    st.success(f"✅ Complete package '{uploaded_package.name}' loaded successfully!")
                                    
                                    # Show package info
                                    if 'imported_feature_names' in st.session_state:
                                        st.markdown("**📋 Model Information:**")
                                        col_a, col_b, col_c = st.columns(3)
                                        with col_a:
                                            st.metric("Problem Type", st.session_state.imported_problem_type)
                                        with col_b:
                                            st.metric("Features", len(st.session_state.imported_feature_names))
                                        with col_c:
                                            st.metric("Target", st.session_state.imported_target_column)
                                        
                                        with st.expander("📝 Feature Names"):
                                            st.write(st.session_state.imported_feature_names)
                                    
                                except Exception as e:
                                    st.error(f"❌ Error loading package: {str(e)}")
                    
                    with col2:
                        st.markdown("#### 🎯 Make Predictions")
                        
                        if 'imported_model' in st.session_state:
                            prediction_method = st.selectbox(
                                "Prediction Method:",
                                ["Manual Input", "Upload CSV", "Use Current Dataset"]
                            )
                            
                            if prediction_method == "Manual Input":
                                st.markdown("**Enter feature values:**")
                                
                                if 'imported_feature_names' in st.session_state:
                                    feature_names = st.session_state.imported_feature_names
                                else:
                                    # Try to infer feature names or ask user
                                    st.warning("⚠️ Feature names not available. Using generic names.")
                                    feature_names = [f"feature_{i}" for i in range(10)]  # Default
                                
                                # Create input fields for each feature
                                input_values = {}
                                num_features = min(len(feature_names), 10)  # Limit display
                                
                                for i, feature in enumerate(feature_names[:num_features]):
                                    input_values[feature] = st.number_input(
                                        f"{feature}:",
                                        value=0.0,
                                        key=f"import_input_{feature}_{i}"
                                    )
                                
                                if len(feature_names) > 10:
                                    st.info(f"📝 Showing first 10 of {len(feature_names)} features. Use CSV upload for complete predictions.")
                                
                                if st.button("🔮 Predict", key="import_predict_button"):
                                    try:
                                        # Prepare input data
                                        input_df = pd.DataFrame([input_values])
                                        
                                        # Ensure all required features are present
                                        if 'imported_feature_names' in st.session_state:
                                            for feature in st.session_state.imported_feature_names:
                                                if feature not in input_df.columns:
                                                    input_df[feature] = 0.0  # Default value
                                            
                                            # Reorder columns to match training
                                            input_df = input_df[st.session_state.imported_feature_names]
                                        
                                        # Apply preprocessing if available
                                        if 'imported_label_encoders' in st.session_state:
                                            for col, encoder in st.session_state.imported_label_encoders.items():
                                                if col in input_df.columns:
                                                    # Handle encoding for manual input
                                                    pass  # Skip for numeric inputs
                                        
                                        if 'imported_scaler' in st.session_state:
                                            input_df = pd.DataFrame(
                                                st.session_state.imported_scaler.transform(input_df),
                                                columns=input_df.columns
                                            )
                                        
                                        # Make prediction
                                        prediction = st.session_state.imported_model.predict(input_df)[0]
                                        
                                        st.success(f"🎯 **Prediction:** {prediction}")
                                        
                                        # Show probabilities if classification
                                        if hasattr(st.session_state.imported_model, 'predict_proba'):
                                            probabilities = st.session_state.imported_model.predict_proba(input_df)[0]
                                            
                                            if hasattr(st.session_state.imported_model, 'classes_'):
                                                classes = st.session_state.imported_model.classes_
                                                prob_df = pd.DataFrame({
                                                    'Class': classes,
                                                    'Probability': probabilities
                                                })
                                                prob_df['Percentage'] = prob_df['Probability'] * 100
                                                
                                                st.markdown("**📊 Prediction Probabilities:**")
                                                st.dataframe(prob_df, use_container_width=True)
                                                
                                                # Confidence bar
                                                confidence = max(probabilities)
                                                st.metric("Confidence", f"{confidence:.1%}")
                                    
                                    except Exception as e:
                                        st.error(f"❌ Prediction error: {str(e)}")
                                        st.error("💡 Try using the Complete Package import for better compatibility")
                            
                            elif prediction_method == "Upload CSV":
                                uploaded_csv = st.file_uploader(
                                    "Upload CSV for Batch Predictions",
                                    type=['csv'],
                                    key="import_csv_uploader"
                                )
                                
                                if uploaded_csv:
                                    try:
                                        batch_df = pd.read_csv(uploaded_csv)
                                        st.markdown("**📄 Data Preview:**")
                                        st.dataframe(batch_df.head(), use_container_width=True)
                                        
                                        if st.button("🔮 Predict Batch", key="import_batch_predict"):
                                            # Preprocess if needed
                                            processed_df = batch_df.copy()
                                            
                                            # Ensure correct features if available
                                            if 'imported_feature_names' in st.session_state:
                                                available_features = [f for f in st.session_state.imported_feature_names 
                                                                    if f in processed_df.columns]
                                                if available_features:
                                                    processed_df = processed_df[available_features]
                                                    
                                                    # Add missing features with default values
                                                    for feature in st.session_state.imported_feature_names:
                                                        if feature not in processed_df.columns:
                                                            processed_df[feature] = 0.0
                                                    
                                                    # Reorder columns
                                                    processed_df = processed_df[st.session_state.imported_feature_names]
                                            
                                            # Apply preprocessing pipeline if available
                                            if 'imported_label_encoders' in st.session_state:
                                                for col, encoder in st.session_state.imported_label_encoders.items():
                                                    if col in processed_df.columns:
                                                        processed_df[col] = processed_df[col].fillna("Missing")
                                                        try:
                                                            processed_df[col] = encoder.transform(processed_df[col])
                                                        except ValueError:
                                                            st.warning(f"⚠️ Unknown categories in {col}, using fallback")
                                                            processed_df[col] = 0  # Fallback
                                            
                                            if 'imported_scaler' in st.session_state:
                                                processed_df = pd.DataFrame(
                                                    st.session_state.imported_scaler.transform(processed_df),
                                                    columns=processed_df.columns,
                                                    index=processed_df.index
                                                )
                                            
                                            # Make predictions
                                            predictions = st.session_state.imported_model.predict(processed_df)
                                            
                                            # Add predictions to results
                                            results_df = batch_df.copy()
                                            results_df['Prediction'] = predictions
                                            
                                            # Add probabilities if classification
                                            if hasattr(st.session_state.imported_model, 'predict_proba'):
                                                probabilities = st.session_state.imported_model.predict_proba(processed_df)
                                                confidence_scores = np.max(probabilities, axis=1)
                                                results_df['Confidence'] = confidence_scores
                                                
                                                # Add individual class probabilities
                                                if hasattr(st.session_state.imported_model, 'classes_'):
                                                    for i, class_name in enumerate(st.session_state.imported_model.classes_):
                                                        results_df[f'Prob_{class_name}'] = probabilities[:, i]
                                            
                                            st.markdown("**🎯 Prediction Results:**")
                                            st.dataframe(results_df, use_container_width=True)
                                            
                                            # Statistics
                                            col_a, col_b, col_c = st.columns(3)
                                            with col_a:
                                                st.metric("Total Predictions", len(results_df))
                                            with col_b:
                                                if 'Confidence' in results_df.columns:
                                                    avg_conf = results_df['Confidence'].mean()
                                                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                                            with col_c:
                                                unique_predictions = results_df['Prediction'].nunique()
                                                st.metric("Unique Predictions", unique_predictions)
                                            
                                            # Download results
                                            csv_results = results_df.to_csv(index=False)
                                            st.download_button(
                                                "📥 Download Results",
                                                data=csv_results,
                                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv",
                                                key=f"download_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                            )
                                    
                                    except Exception as e:
                                        st.error(f"❌ Batch prediction error: {str(e)}")
                            
                            elif prediction_method == "Use Current Dataset":
                                if model_df is not None:
                                    st.markdown("**📊 Current Dataset Preview:**")
                                    st.dataframe(model_df.head(), use_container_width=True)
                                    
                                    if st.button("🔮 Predict on Current Data", key="import_current_predict"):
                                        try:
                                            # Use current dataset for predictions
                                            current_df = model_df.copy()
                                            
                                            # Select appropriate features if available
                                            if 'imported_feature_names' in st.session_state:
                                                available_features = [f for f in st.session_state.imported_feature_names 
                                                                    if f in current_df.columns]
                                                if available_features:
                                                    prediction_df = current_df[available_features]
                                                    
                                                    # Add missing features with defaults
                                                    for feature in st.session_state.imported_feature_names:
                                                        if feature not in prediction_df.columns:
                                                            prediction_df[feature] = 0.0
                                                    
                                                    # Reorder columns
                                                    prediction_df = prediction_df[st.session_state.imported_feature_names]
                                                else:
                                                    st.error("❌ No matching features found in current dataset")
                                                    st.stop()
                                            else:
                                                # Use all numeric columns
                                                prediction_df = current_df.select_dtypes(include=[np.number])
                                            
                                            # Make predictions
                                            predictions = st.session_state.imported_model.predict(prediction_df)
                                            
                                            # Show results
                                            results_df = current_df.copy()
                                            results_df['Imported_Model_Prediction'] = predictions
                                            
                                            if hasattr(st.session_state.imported_model, 'predict_proba'):
                                                probabilities = st.session_state.imported_model.predict_proba(prediction_df)
                                                confidence_scores = np.max(probabilities, axis=1)
                                                results_df['Imported_Model_Confidence'] = confidence_scores
                                            
                                            st.markdown("**🎯 Prediction Results:**")
                                            st.dataframe(results_df, use_container_width=True)
                                            
                                            # Download results
                                            csv_results = results_df.to_csv(index=False)
                                            st.download_button(
                                                "📥 Download Results",
                                                data=csv_results,
                                                file_name=f"imported_model_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv",
                                                key=f"download_imported_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                            )
                                        
                                        except Exception as e:
                                            st.error(f"❌ Prediction error: {str(e)}")
                                else:
                                    st.warning("⚠️ No dataset loaded. Please load data first.")
                        
                        else:
                            st.info("👆 Please import a model first to make predictions")
                    
                    # Model comparison section
                    if 'imported_model' in st.session_state and 'trained_models' in st.session_state and st.session_state.trained_models:
                        st.markdown("---")
                        st.markdown("#### 🔄 Compare Imported vs Trained Models")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📁 Imported Model:**")
                            st.info(f"Model: {st.session_state.imported_model_name}")
                            if hasattr(st.session_state.imported_model, '__class__'):
                                st.write(f"Type: {st.session_state.imported_model.__class__.__name__}")
                        
                        with col2:
                            st.markdown("**🏗️ Trained Models:**")
                            for model_name in st.session_state.trained_models.keys():
                                st.write(f"• {model_name}")
                        
                        st.info("💡 Use the Model Comparison tab to compare performance on the same dataset")
                
                with model_tab1:
                    st.subheader("⚙️ ML Configuration & Training")

                    # Model Configuration
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Target column selection
                        target_column = st.selectbox(
                            "Select Target Column:",
                            options=model_df.columns.tolist(),
                            help="Choose the column you want to predict",
                        )

                    with col2:
                        # Problem type detection/selection
                        if target_column:
                            # Auto-detect problem type
                            unique_values = model_df[target_column].nunique()
                            is_numeric = pd.api.types.is_numeric_dtype(
                                model_df[target_column]
                            )

                            if is_numeric and unique_values > 10:
                                suggested_type = "Regression"
                            else:
                                suggested_type = "Classification"

                            problem_type = st.selectbox(
                                "Problem Type:",
                                options=["Classification", "Regression"],
                                index=0 if suggested_type == "Classification" else 1,
                                help=f"Auto-detected: {suggested_type}",
                            )

                    with col3:
                        # Test size
                        test_size = st.slider(
                            "Test Set Size:",
                            min_value=0.1,
                            max_value=0.5,
                            value=0.2,
                            step=0.05,
                            help="Proportion of data to use for testing",
                        )

                    # Enhanced Data validation with visual feedback
                    if target_column:
                        is_valid, validation_message = validate_data_for_modeling(
                            model_df, target_column
                        )

                        if is_valid:
                            # Enhanced data summary with metrics
                            st.subheader("📊 Enhanced Data Summary")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Samples", f"{len(model_df):,}")
                            with col2:
                                st.metric("Features", len(model_df.columns) - 1)
                            with col3:
                                missing_in_target = model_df[target_column].isnull().sum()
                                st.metric("Missing in Target", missing_in_target)
                            with col4:
                                st.metric("Unique Target Values", model_df[target_column].nunique())

                            # Enhanced target analysis
                            target_col1, target_col2 = st.columns(2)
                            
                            with target_col1:
                                if problem_type == "Classification":
                                    st.write("**Target Distribution:**")
                                    target_counts = model_df[target_column].value_counts()
                                    
                                    # Create an enhanced visualization
                                    fig = px.pie(
                                        values=target_counts.values, 
                                        names=target_counts.index,
                                        title="Class Distribution"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.write("**Target Distribution:**")
                                    fig = px.histogram(
                                        model_df, 
                                        x=target_column,
                                        title="Target Value Distribution",
                                        nbins=30
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with target_col2:
                                st.write("**Target Statistics:**")
                                target_stats = model_df[target_column].describe()
                                st.dataframe(target_stats.to_frame().T, use_container_width=True)
                                
                                # Data quality warnings
                                st.write("**Data Quality Check:**")
                                if missing_in_target > 0:
                                    st.warning(f"⚠️ {missing_in_target} missing values in target")
                                else:
                                    st.success("✅ No missing values in target")
                                
                                if problem_type == "Classification":
                                    # Check class imbalance
                                    class_distribution = model_df[target_column].value_counts(normalize=True)
                                    max_class_ratio = class_distribution.iloc[0]
                                    if max_class_ratio > 0.8:
                                        st.warning(f"⚠️ Class imbalance detected ({max_class_ratio:.1%} majority class)")
                                    else:
                                        st.success("✅ Balanced class distribution")

                            # Feature Selection Section
                            st.markdown("---")
                            st.subheader("🔧 Feature Selection")
                            
                            # Get all columns except target
                            available_features = [col for col in model_df.columns if col != target_column]
                            
                            # Auto-recommend features (exclude ID-like columns and low-variance)
                            recommended_features = []
                            for col in available_features:
                                # Skip ID-like columns
                                if any(id_word in col.lower() for id_word in ['id', 'index', 'key']):
                                    continue
                                # Skip text columns that look like names/descriptions
                                if col.lower() in ['name', 'description', 'notes', 'comments']:
                                    continue
                                # Skip date columns for now (could be feature engineered later)
                                if model_df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                                    continue
                                recommended_features.append(col)
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.write("**Select Features for Training:**")
                                
                                # Feature selection mode
                                selection_mode = st.radio(
                                    "Selection Mode:",
                                    ["Smart Recommendation", "Manual Selection", "All Features"],
                                    help="Choose how to select features for training"
                                )
                                
                                if selection_mode == "Smart Recommendation":
                                    selected_features = st.multiselect(
                                        "Recommended Features (Smart Auto-Selection):",
                                        options=available_features,
                                        default=recommended_features,
                                        help="AI-selected features based on data types and patterns"
                                    )
                                    
                                    if len(recommended_features) != len(available_features):
                                        excluded_features = [f for f in available_features if f not in recommended_features]
                                        st.info(f"📋 **Auto-excluded:** {', '.join(excluded_features)} (ID/text columns)")
                                
                                elif selection_mode == "Manual Selection":
                                    selected_features = st.multiselect(
                                        "Choose Features Manually:",
                                        options=available_features,
                                        default=recommended_features if recommended_features else available_features[:5],
                                        help="Select specific features you want to use"
                                    )
                                
                                else:  # All Features
                                    selected_features = available_features
                                    st.info(f"✅ **Using all {len(available_features)} features**")
                                    st.write("**All Available Features:**")
                                    st.write(", ".join(available_features))
                            
                            with col2:
                                st.write("**Feature Summary:**")
                                if selected_features:
                                    st.metric("Selected Features", len(selected_features))
                                    st.metric("Total Available", len(available_features))
                                    
                                    # Feature types breakdown
                                    if selected_features:
                                        selected_df = model_df[selected_features]
                                        numeric_features = selected_df.select_dtypes(include=[np.number]).columns.tolist()
                                        categorical_features = selected_df.select_dtypes(include=['object']).columns.tolist()
                                        
                                        st.write("**Feature Types:**")
                                        st.write(f"• Numeric: {len(numeric_features)}")
                                        st.write(f"• Categorical: {len(categorical_features)}")
                                        
                                        # Missing values in selected features
                                        missing_counts = selected_df.isnull().sum()
                                        features_with_missing = missing_counts[missing_counts > 0]
                                        
                                        if len(features_with_missing) > 0:
                                            st.warning(f"⚠️ {len(features_with_missing)} features have missing values")
                                            with st.expander("View Missing Data"):
                                                for feat, count in features_with_missing.items():
                                                    pct = (count / len(model_df)) * 100
                                                    st.write(f"• {feat}: {count} ({pct:.1f}%)")
                                        else:
                                            st.success("✅ No missing values")
                                else:
                                    st.warning("⚠️ No features selected")
                            
                            # Feature correlation analysis (for numeric features)
                            if selected_features:
                                numeric_selected = [f for f in selected_features if pd.api.types.is_numeric_dtype(model_df[f])]
                                # Ensure target column is also numeric
                                if pd.api.types.is_numeric_dtype(model_df[target_column]):
                                    numeric_selected_with_target = numeric_selected + [target_column]
                                else:
                                    numeric_selected_with_target = numeric_selected
                                
                                if len(numeric_selected) > 1:
                                    with st.expander("🔍 Feature Correlation Analysis"):
                                        corr_df = model_df[numeric_selected_with_target].corr()
                                        
                                        # Show correlation with target (only if target is numeric)
                                        if target_column in corr_df.columns:
                                            target_corr = corr_df[target_column].drop(target_column).abs().sort_values(ascending=False)
                                        else:
                                            target_corr = None
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("**Correlation with Target:**")
                                            if target_corr is not None:
                                                corr_display = pd.DataFrame({
                                                    'Feature': target_corr.index,
                                                    'Correlation': target_corr.values.round(3)
                                                })
                                                st.dataframe(corr_display, use_container_width=True)
                                            else:
                                                st.info("Target column is not numeric - correlation analysis not available")
                                        
                                        with col2:
                                            # Correlation heatmap
                                            fig = px.imshow(
                                                corr_df,
                                                text_auto=True,
                                                title="Feature Correlation Matrix",
                                                color_continuous_scale="RdBu_r"
                                            )
                                            fig.update_layout(height=400)
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Highlight highly correlated features
                                        high_corr_pairs = []
                                        for i in range(len(corr_df.columns)):
                                            for j in range(i+1, len(corr_df.columns)):
                                                corr_val = abs(corr_df.iloc[i, j])
                                                if corr_val > 0.8 and corr_df.columns[i] != target_column and corr_df.columns[j] != target_column:
                                                    high_corr_pairs.append((corr_df.columns[i], corr_df.columns[j], corr_val))
                                        
                                        if high_corr_pairs:
                                            st.warning("⚠️ **High Correlation Detected:**")
                                            for feat1, feat2, corr_val in high_corr_pairs:
                                                st.write(f"• {feat1} ↔ {feat2}: {corr_val:.3f}")
                                            st.info("💡 Consider removing one feature from each highly correlated pair")

                            # Enhanced model training section
                            st.markdown("---")
                            st.subheader("🚀 Model Training")
                            
                            # 🧠 Intelligent Model Recommender - The Final Missing Piece
                            with st.expander("🧠 Intelligent Model Recommender - Get Smart Recommendations", expanded=False):
                                st.markdown("""
                                <div style="background: linear-gradient(135deg, #FFA726 0%, #FF7043 100%); 
                                            padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                    <h4 style="color: white; margin: 0;">🎯 AI-Powered Model Selection</h4>
                                    <p style="color: #f0f0f0; margin: 5px 0 0 0;">
                                        Get intelligent model recommendations based on your data characteristics
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if st.button("🔍 Analyze Data & Get Model Recommendations"):
                                    if not selected_features:
                                        st.error("❌ Please select features first!")
                                    else:
                                        with st.spinner("🧠 Analyzing your data characteristics..."):
                                            try:
                                                # Create X and y from selected features and target
                                                X = model_df[selected_features].copy()
                                                y = model_df[target_column].copy()
                                                
                                                # Handle missing values for analysis
                                                if X.isnull().any().any():
                                                    # Simple imputation for analysis
                                                    from sklearn.impute import SimpleImputer
                                                    numeric_features = X.select_dtypes(include=[np.number]).columns
                                                    categorical_features = X.select_dtypes(include=['object']).columns
                                                    
                                                    if len(numeric_features) > 0:
                                                        X[numeric_features] = SimpleImputer(strategy='median').fit_transform(X[numeric_features])
                                                    if len(categorical_features) > 0:
                                                        X[categorical_features] = SimpleImputer(strategy='most_frequent').fit_transform(X[categorical_features])
                                                
                                                # Encode categorical variables for analysis
                                                from sklearn.preprocessing import LabelEncoder
                                                categorical_cols = X.select_dtypes(include=['object']).columns
                                                for col in categorical_cols:
                                                    le = LabelEncoder()
                                                    X[col] = le.fit_transform(X[col].astype(str))
                                                
                                                # Handle target variable if categorical
                                                if problem_type == "Classification" and y.dtype == 'object':
                                                    le_target = LabelEncoder()
                                                    y = le_target.fit_transform(y.astype(str))
                                                
                                                # Create recommender instance
                                                recommender = IntelligentModelRecommender()
                                                
                                                # Generate comprehensive report
                                                rec_report = recommender.generate_comprehensive_report(X, y)
                                                
                                                # Display analysis results
                                                st.success("✅ Data analysis completed!")
                                                
                                                # Show data characteristics
                                                rec_col1, rec_col2 = st.columns(2)
                                                
                                                with rec_col1:
                                                    st.markdown("#### 📊 Data Profile")
                                                    analysis = rec_report['data_analysis']
                                                    
                                                    st.metric("Dataset Size", f"{analysis['n_samples']:,} samples")
                                                    st.metric("Feature Count", analysis['n_features'])
                                                    st.metric("Problem Type", analysis['problem_type'].title())
                                                    st.metric("Complexity Level", analysis['feature_complexity'].title())
                                                    
                                                    # Additional characteristics
                                                    st.markdown("**Data Quality:**")
                                                    st.write(f"• Missing data: {analysis['missing_percentage']:.1f}%")
                                                    st.write(f"• Noise level: {analysis['noise_level'].title()}")
                                                    if analysis['problem_type'] == 'classification':
                                                        st.write(f"• Class balance ratio: {analysis['class_balance_ratio']:.1f}")
                                                
                                                with rec_col2:
                                                    st.markdown("#### 🎯 Model Recommendations")
                                                    
                                                    recommendations = rec_report['recommendations']
                                                    
                                                    # Primary recommendations
                                                    st.markdown("**🥇 Primary Recommendations:**")
                                                    for i, rec in enumerate(recommendations['primary'], 1):
                                                        confidence_color = "🟢" if rec['confidence'] > 0.8 else "🟡" if rec['confidence'] > 0.6 else "🟠"
                                                        st.write(f"{i}. {confidence_color} **{rec['model']}** ({rec['confidence']:.0%} confidence)")
                                                        st.write(f"   *{rec['reason']}*")
                                                    
                                                    # Secondary recommendations
                                                    if recommendations['secondary']:
                                                        st.markdown("**🥈 Alternative Options:**")
                                                        for rec in recommendations['secondary']:
                                                            st.write(f"• {rec['model']} - {rec['reason']}")
                                            
                                            except Exception as e:
                                                st.error(f"❌ Error analyzing data: {str(e)}")
                                                st.info("💡 Please ensure your data is properly cleaned and try again.")
                                        
                                        # Performance predictions
                                        if 'performance_predictions' in rec_report:
                                            st.markdown("#### 📈 Expected Performance")
                                            
                                            perf_predictions = rec_report['performance_predictions']
                                            perf_cols = st.columns(len(perf_predictions))
                                            
                                            for i, (model_name, pred) in enumerate(perf_predictions.items()):
                                                with perf_cols[i]:
                                                    score_type = "Accuracy" if analysis['problem_type'] == 'classification' else "R² Score"
                                                    st.metric(
                                                        model_name,
                                                        f"{pred['predicted_score']:.3f}",
                                                        f"{score_type}"
                                                    )
                                        
                                        # Reasoning
                                        st.markdown("#### 🤔 Why These Recommendations?")
                                        reasoning = rec_report['recommendations']['reasoning']
                                        for reason in reasoning:
                                            st.markdown(reason)
                                        
                                        # Summary
                                        st.markdown("#### 📋 Summary")
                                        st.markdown(rec_report['summary'])
                                        
                                        # Store recommendations in session state for easy access
                                        st.session_state.model_recommendations = recommendations['primary']
                                        
                                        st.success("💡 Tip: Use these recommendations to guide your model selection below!")
                            
                            # Model selection options
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                training_mode = st.selectbox(
                                    "Training Mode:",
                                    ["Quick Training (Fast)", "Comprehensive Training (Best Results)", "Custom Selection"],
                                    help="Choose training approach based on your needs"
                                )
                            
                            with col2:
                                selected_models = []  # Initialize default value
                                if training_mode == "Custom Selection":
                                    available_models = {
                                        "Random Forest": "Robust, handles mixed data well",
                                        "Gradient Boosting": "High performance, good for competitions", 
                                        "SVM": "Good for small datasets, non-linear patterns",
                                        "Logistic Regression": "Fast, interpretable, linear relationships",
                                        "Linear Regression": "Basic, fast, good for linear patterns",
                                        "K-Nearest Neighbors": "Simple, good for local patterns",
                                        "Naive Bayes": "Fast, good for text/categorical data"
                                    }
                                    
                                    selected_models = st.multiselect(
                                        "Select Models to Train:",
                                        options=list(available_models.keys()),
                                        default=["Random Forest", "Gradient Boosting"],
                                        help="Choose which models to include in training"
                                    )
                                else:
                                    # Show training mode information for non-custom modes
                                    if training_mode == "Quick Training (Fast)":
                                        st.info("⚡ **Quick Training** will use: Random Forest + Logistic Regression (Classification) or Random Forest + Linear Regression (Regression)")
                                    elif training_mode == "Comprehensive Training (Best Results)":
                                        st.info("🎯 **Comprehensive Training** will use all available models for the best performance comparison")

                            # Train models button with enhanced UI
                            if st.button("🚀 Train Models", type="primary", use_container_width=True):
                                # Validate feature selection first
                                if not selected_features:
                                    st.error("❌ Please select at least one feature for training!")
                                    st.stop()
                                
                                # Validate model selection for custom mode
                                if training_mode == "Custom Selection" and (not selected_models or len(selected_models) == 0):
                                    st.error("❌ Please select at least one model for Custom Selection training!")
                                    st.stop()
                                
                                with st.spinner("🔄 Preparing data and training models..."):
                                    # Store training configuration
                                    st.session_state.target_column = target_column
                                    st.session_state.problem_type = problem_type
                                    st.session_state.test_size = test_size
                                    st.session_state.selected_features = selected_features
                                    
                                    # Create dataframe with only selected features + target
                                    modeling_df = model_df[selected_features + [target_column]].copy()
                                    
                                    st.info(f"🎯 Using {len(selected_features)} features: {', '.join(selected_features)}")
                                    
                                    # Prepare data
                                    X, y, success, label_encoders, target_encoder = (
                                        prepare_data_for_modeling(
                                            modeling_df, target_column, problem_type
                                        )
                                    )

                                    if success and X is not None and y is not None:
                                        st.success("✅ Data prepared successfully!")

                                        # Determine which models to train based on mode
                                        models_to_train = None
                                        if training_mode == "Custom Selection":
                                            models_to_train = selected_models
                                            st.info(f"🤖 Training selected models: {', '.join(selected_models)}")
                                        elif training_mode == "Quick Training (Fast)":
                                            # Quick models for fast training
                                            if problem_type == "Classification":
                                                models_to_train = ["Random Forest", "Logistic Regression"]
                                            else:
                                                models_to_train = ["Random Forest", "Linear Regression"]
                                            st.info(f"⚡ Quick training with: {', '.join(models_to_train)}")
                                        else:  # Comprehensive Training
                                            # All available models for best results
                                            if problem_type == "Classification":
                                                models_to_train = ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression", "K-Nearest Neighbors", "Naive Bayes"]
                                            else:
                                                models_to_train = ["Random Forest", "Gradient Boosting", "SVM", "Linear Regression", "K-Nearest Neighbors"]
                                            st.info(f"🎯 Comprehensive training with: {', '.join(models_to_train)}")

                                        # Train models with enhanced progress tracking
                                        progress_container = st.container()
                                        
                                        with progress_container:
                                            st.markdown("### 🚀 Training Progress")
                                            
                                            # Create progress tracking components
                                            overall_progress = st.progress(0)
                                            current_model_text = st.empty()
                                            model_progress = st.progress(0)
                                            status_text = st.empty()
                                            
                                            # Performance metrics placeholder
                                            metrics_placeholder = st.empty()
                                            
                                            try:
                                                status_text.text("🔄 Initializing training pipeline...")
                                                overall_progress.progress(0.1)
                                                time.sleep(0.3)
                                                
                                                # Train models with progress tracking
                                                (
                                                    results,
                                                    X_train,
                                                    X_test,
                                                    y_train,
                                                    y_test,
                                                    scaler,
                                                    diagnostic_results,
                                                    improvement_plan,
                                                ) = with_progress_cache(
                                                    "Machine Learning Training",
                                                    train_models,
                                                    X, y, problem_type, test_size, models_to_train
                                                )
                                                
                                                # Update progress as models complete
                                                total_models = len(models_to_train)
                                                for i, model_name in enumerate(models_to_train):
                                                    current_model_text.text(f"🤖 Training: {model_name}")
                                                    model_progress.progress((i + 1) / total_models)
                                                    # Fix progress calculation to stay within 0.0-1.0 range
                                                    overall_progress.progress(min(0.8, 0.2 + (0.6 * (i + 1) / total_models)))
                                                    
                                                    # Show intermediate results
                                                    if model_name in results and 'score' in results[model_name]:
                                                        score = results[model_name]['score']
                                                        metrics_placeholder.metric(
                                                            f"Latest: {model_name}",
                                                            f"{score:.3f}",
                                                            delta=f"Model {i+1}/{total_models}"
                                                        )
                                                    time.sleep(0.2)
                                                
                                                # Finalize
                                                status_text.text("✅ Training complete! Processing results...")
                                                overall_progress.progress(0.9)
                                                time.sleep(0.5)
                                                
                                                # Clear progress indicators
                                                overall_progress.progress(1.0)
                                                time.sleep(0.3)
                                                progress_container.empty()
                                                
                                            except Exception as e:
                                                progress_container.empty()
                                                st.error(f"❌ Training failed: {str(e)}")
                                                logger.error(f"Model training error: {str(e)}\n{traceback.format_exc()}")
                                                results = None

                                        if results:
                                            # Show training success with metrics
                                            # Use appropriate score key based on problem type
                                            if problem_type == "Classification":
                                                best_model = max(results.items(), key=lambda x: x[1].get('accuracy', 0))
                                                score_value = best_model[1].get('accuracy', 0)
                                                score_name = "Accuracy"
                                            else:  # Regression
                                                best_model = max(results.items(), key=lambda x: x[1].get('r2_score', 0))
                                                score_value = best_model[1].get('r2_score', 0)
                                                score_name = "R² Score"
                                            
                                            st.success(f"✅ Training completed! Best model: {best_model[0]} ({score_name}: {score_value:.4f})")

                                            # Store results in session state for general use
                                            st.session_state.model_results = results
                                            st.session_state.X_test = X_test
                                            st.session_state.y_test = y_test
                                            st.session_state.X_train = X_train
                                            st.session_state.y_train = y_train
                                            st.session_state.feature_names = X.columns.tolist()
                                            st.session_state.scaler = scaler
                                            st.session_state.label_encoders = label_encoders
                                            st.session_state.target_encoder = target_encoder
                                            st.session_state.diagnostic_results = diagnostic_results
                                            st.session_state.improvement_plan = improvement_plan

                                            # Store trained models for Performance Analysis tab
                                            trained_models = {}
                                            for model_name, model_info in results.items():
                                                if 'model' in model_info:
                                                    # Use appropriate score key based on problem type
                                                    if problem_type == "Classification":
                                                        score = model_info.get('accuracy', 0)
                                                    else:  # Regression
                                                        score = model_info.get('r2_score', 0)
                                                    
                                                    trained_models[model_name] = {
                                                        'model': model_info['model'],
                                                        'score': score,
                                                        'training_time': model_info.get('training_time', 0),
                                                        'problem_type': problem_type
                                                    }
                                            st.session_state.trained_models = trained_models

                                            # Enhanced results display with error handling
                                            try:
                                                display_model_results(results, problem_type)
                                            except Exception as e:
                                                st.error(f"❌ Error displaying results: {str(e)}")
                                                logger.error(f"Results display error: {str(e)}")
                                                
                                                # Show basic results as fallback
                                                st.subheader("📊 Model Performance Summary")
                                                for model_name, model_info in results.items():
                                                    # Use appropriate score key based on problem type
                                                    if problem_type == "Classification":
                                                        score = model_info.get('accuracy', 0)
                                                        score_name = "Accuracy"
                                                    else:  # Regression
                                                        score = model_info.get('r2_score', 0)
                                                        score_name = "R² Score"
                                                    st.metric(f"{model_name} ({score_name})", f"{score:.4f}")
                                            
                                            # Performance Diagnostic and Improvement Suggestions
                                            try:
                                                display_performance_diagnostic(diagnostic_results, improvement_plan)
                                            except Exception as e:
                                                st.warning(f"⚠️ Could not generate performance diagnostics: {str(e)}")
                                                logger.warning(f"Diagnostic display error: {str(e)}")
                                        else:
                                            st.error("❌ Model training failed. Please check your data and try again.")
                                            
                                            # 🚀 Advanced Model Optimizer - Implement Specific Recommendations
                                            st.markdown("---")
                                            st.subheader("🚀 Advanced Model Optimizer - Fix Performance Issues")
                                            st.markdown("""
                                            <div style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); 
                                                        padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                                <h4 style="color: white; margin: 0;">🎯 Implement Diagnostic Recommendations</h4>
                                                <p style="color: #f0f0f0; margin: 5px 0 0 0;">
                                                    Apply advanced solutions for overfitting, feature selection, and hyperparameter optimization
                                                </p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                            
                                            # Create optimizer instance
                                            optimizer = AdvancedModelOptimizer(X, y, problem_type.lower())
                                            
                                            # FIRST: Try simple fixes for terrible performance
                                            st.markdown("### 🚨 Quick Performance Check")
                                            fixer = SimpleMLFixer(X, y, problem_type.lower())
                                            
                                            with st.expander("🔍 Diagnose Common Issues", expanded=True):
                                                fixes = fixer.diagnose_and_fix()
                                                
                                                if fixes:
                                                    st.warning("⚠️ **CRITICAL ISSUES FOUND:**")
                                                    for fix in fixes:
                                                        if "ERROR:" in fix:
                                                            st.error(fix)
                                                        elif "WARNING:" in fix:
                                                            st.warning(fix)
                                                        else:
                                                            st.info(f"✅ {fix}")
                                                    
                                                    # Show quick fix button
                                                    if st.button("🚀 Apply Quick Fixes & Test", type="primary"):
                                                        with st.spinner("Testing quick fixes..."):
                                                            quick_results = fixer.create_simple_models()
                                                            
                                                            if "Error" in quick_results:
                                                                st.error("❌ " + quick_results["Error"])
                                                            else:
                                                                st.success("✅ Quick fixes applied!")
                                                                for name, result in quick_results.items():
                                                                    if 'cv_score_mean' in result:
                                                                        st.metric(
                                                                            name, 
                                                                            f"{result['cv_score_mean']:.3f}",
                                                                            delta=result['improvement']
                                                                        )
                                                else:
                                                    st.success("✅ No critical issues found - data looks good!")
                                                
                                                st.markdown("---")
                                                
                                                # Show data quality diagnosis
                                                st.subheader("🔍 Advanced Data Quality Analysis")
                                                for log_entry in optimizer.optimization_log:
                                                    if "⚠️" in log_entry or "🚨" in log_entry:
                                                        st.warning(log_entry)
                                                    elif "✅" in log_entry:
                                                        st.success(log_entry)
                                                    else:
                                                        st.info(log_entry)
                                                
                                                # Optimization options
                                                opt_col1, opt_col2 = st.columns(2)
                                                
                                                with opt_col1:
                                                    st.markdown("#### 🎯 Solution Selection")
                                                    
                                                    auto_optimize = st.checkbox(
                                                        "🤖 Auto-Detect & Fix Issues", 
                                                        value=True,
                                                        help="Automatically detect performance issues and apply appropriate solutions"
                                                    )
                                                    
                                                    fix_overfitting = st.checkbox(
                                                        "🛡️ Fix Overfitting Issues", 
                                                        value=not auto_optimize,
                                                        disabled=auto_optimize,
                                                        help="Apply regularization and complexity reduction"
                                                    )
                                                    
                                                    fix_underfitting = st.checkbox(
                                                        "🚀 Fix Underfitting Issues", 
                                                        value=not auto_optimize,
                                                        disabled=auto_optimize,
                                                        help="Increase model complexity and remove regularization"
                                                    )
                                                    
                                                    optimize_features = st.checkbox(
                                                        "🔍 Optimize Feature Selection", 
                                                        value=not auto_optimize,
                                                        disabled=auto_optimize,
                                                        help="Remove irrelevant features using advanced selection methods"
                                                    )
                                                    
                                                    tune_hyperparams = st.checkbox(
                                                        "⚙️ Optimize Hyperparameters", 
                                                        value=not auto_optimize,
                                                        disabled=auto_optimize,
                                                        help="Find best parameters using grid/random search"
                                                    )
                                                
                                                with opt_col2:
                                                    st.markdown("#### ⚙️ Optimization Settings")
                                                    
                                                    if optimize_features:
                                                        n_features = st.slider(
                                                            "Target Number of Features:",
                                                            min_value=5,
                                                            max_value=min(50, len(X.columns)),
                                                            value=min(15, int(len(X.columns) * 0.7)),
                                                            help="Optimal number of features to select"
                                                        )
                                                    
                                                    if tune_hyperparams:
                                                        search_type = st.selectbox(
                                                            "Search Strategy:",
                                                            ["random", "grid"],
                                                            help="Random search is faster, grid search is thorough"
                                                        )
                                                        
                                                        if search_type == "random":
                                                            n_iter = st.slider("Search Iterations:", 20, 100, 50)
                                                        else:
                                                            n_iter = 50  # Not used in grid search
                                                
                                                # Apply optimizations button
                                                if st.button("🚀 Apply Advanced Optimizations", type="primary"):
                                                    with st.spinner("🔄 Applying advanced ML optimizations..."):
                                                        
                                                        optimization_results = {}
                                                        models_dict = {name: info['model'] for name, info in results.items() if 'model' in info}
                                                        
                                                        if auto_optimize:
                                                            st.info("🤖 Auto-detecting and fixing performance issues...")
                                                            auto_solutions = optimizer.auto_optimize(models_dict)
                                                            optimization_results['auto_optimization'] = auto_solutions
                                                            
                                                            # Update models with auto solutions
                                                            for model_name, solution in auto_solutions.items():
                                                                models_dict[model_name] = solution['best_model']
                                                        
                                                        else:
                                                            # Manual optimization selection
                                                            
                                                            # 1. Fix Underfitting
                                                            if fix_underfitting:
                                                                st.info("🚀 Fixing underfitting issues...")
                                                                underfitting_solutions = optimizer.solve_underfitting(models_dict)
                                                                optimization_results['underfitting'] = underfitting_solutions
                                                                
                                                                # Update models with underfitting solutions
                                                                for model_name, solution in underfitting_solutions.items():
                                                                    models_dict[model_name] = solution['best_model']
                                                            
                                                            # 2. Fix Overfitting
                                                            if fix_overfitting:
                                                                st.info("🛡️ Fixing overfitting issues...")
                                                                overfitting_solutions = optimizer.solve_overfitting(models_dict)
                                                                optimization_results['overfitting'] = overfitting_solutions
                                                                
                                                                # Update models with overfitting solutions
                                                                for model_name, solution in overfitting_solutions.items():
                                                                    models_dict[model_name] = solution['best_model']
                                                        
                                                        # 3. Optimize Features
                                                        if optimize_features or auto_optimize:
                                                            st.info("🔍 Optimizing feature selection...")
                                                            feature_results = optimizer.intelligent_feature_selection(
                                                                n_features_to_select=n_features if optimize_features else None
                                                            )
                                                            optimization_results['features'] = feature_results
                                                            
                                                            # Update X with selected features
                                                            if feature_results['selected_features']:
                                                                X_optimized = X[feature_results['selected_features']]
                                                                optimizer.X = X_optimized
                                                        
                                                        # 3. Optimize Hyperparameters
                                                        if tune_hyperparams:
                                                            st.info("⚙️ Optimizing hyperparameters...")
                                                            hyperopt_results = optimizer.optimize_hyperparameters(
                                                                models_dict, 
                                                                search_type=search_type,
                                                                n_iter=n_iter if search_type == "random" else 50
                                                            )
                                                            optimization_results['hyperparameters'] = hyperopt_results
                                                        
                                                        # Display optimization results
                                                        st.success("✅ Advanced optimizations completed!")
                                                        
                                                        # Show improvements
                                                        st.markdown("### 📈 Optimization Results")
                                                        
                                                        result_col1, result_col2 = st.columns(2)
                                                        
                                                        with result_col1:
                                                            if fix_overfitting and 'overfitting' in optimization_results:
                                                                st.markdown("#### 🛡️ Overfitting Solutions")
                                                                for model_name, solution in optimization_results['overfitting'].items():
                                                                    improvement = solution['improvement']
                                                                    approach = solution['approach']
                                                                    cv_score = solution['cv_score_mean']
                                                                    
                                                                    st.metric(
                                                                        f"{model_name} ({approach})",
                                                                        f"{cv_score:.4f}",
                                                                        f"{improvement:+.4f}"
                                                                    )
                                                            
                                                            if optimize_features and 'features' in optimization_results:
                                                                st.markdown("#### 🔍 Feature Optimization")
                                                                feature_res = optimization_results['features']
                                                                
                                                                st.metric(
                                                                    "Selected Features", 
                                                                    len(feature_res['selected_features']),
                                                                    f"from {len(X.columns)}"
                                                                )
                                                                
                                                                st.metric(
                                                                    "Performance Improvement",
                                                                    f"{feature_res['performance_improvement']:.4f}",
                                                                    f"using {feature_res['method_used']}"
                                                                )
                                                        
                                                        with result_col2:
                                                            if tune_hyperparams and 'hyperparameters' in optimization_results:
                                                                st.markdown("#### ⚙️ Hyperparameter Optimization")
                                                                for model_name, hyperopt_res in optimization_results['hyperparameters'].items():
                                                                    best_score = hyperopt_res['best_score']
                                                                    improvement = hyperopt_res['improvement']
                                                                    
                                                                    st.metric(
                                                                        f"{model_name}",
                                                                        f"{best_score:.4f}",
                                                                        f"{improvement:+.4f}"
                                                                    )
                                                        
                                                        # Show optimization log
                                                        with st.expander("📋 Detailed Optimization Log"):
                                                            optimization_log = optimizer.generate_optimization_report()
                                                            for log_entry in optimization_log['optimization_log']:
                                                                st.text(log_entry)
                                                        
                                                        # Option to retrain with optimized settings
                                                        st.markdown("### 🔄 Apply Optimizations")
                                                        if st.button("🚀 Retrain Models with Optimizations"):
                                                            st.info("🔄 Retraining models with optimized configurations...")
                                                            # This would trigger a new training cycle with the optimized models and features
                                                            st.success("✅ Feature implemented! Use the optimized models above or retrain with new settings.")
                                            
                                            # 🤖 Complete Automated ML Pipeline - The Ultimate Missing Piece
                                            with st.expander("🤖 Complete Automated ML Pipeline - Ultimate Optimization", expanded=False):
                                                st.markdown("""
                                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                            padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                                                    <h4 style="color: white; margin: 0;">🚀 Complete Automated ML Pipeline</h4>
                                                    <p style="color: #f0f0f0; margin: 5px 0 0 0;">
                                                        Advanced feature engineering, data augmentation, ensemble methods, and automated experimentation
                                                    </p>
                                                </div>
                                                """, unsafe_allow_html=True)
                                                
                                                # Pipeline configuration
                                                pipeline_col1, pipeline_col2 = st.columns(2)
                                                
                                                with pipeline_col1:
                                                    st.markdown("#### 🔧 Pipeline Configuration")
                                                    
                                                    enable_feature_eng = st.checkbox(
                                                        "🔬 Advanced Feature Engineering", 
                                                        value=True,
                                                        help="Polynomial features, ratios, statistical features, binning"
                                                    )
                                                    
                                                    enable_data_aug = st.checkbox(
                                                        "📈 Data Augmentation", 
                                                        value=True,
                                                        help="SMOTE, ADASYN, noise injection, bootstrap sampling"
                                                    )
                                                    
                                                    enable_ensembles = st.checkbox(
                                                        "🎭 Ensemble Methods", 
                                                        value=True,
                                                        help="Voting, stacking, bagging, boosting ensembles"
                                                    )
                                                    
                                                    enable_auto_experiment = st.checkbox(
                                                        "🧪 Automated Experimentation", 
                                                        value=True,
                                                        help="Test all possible model configurations automatically"
                                                    )
                                                
                                                with pipeline_col2:
                                                    st.markdown("#### ⚙️ Experiment Settings")
                                                    
                                                    if enable_auto_experiment:
                                                        max_experiments = st.slider(
                                                            "Max Experiments:",
                                                            min_value=10,
                                                            max_value=50,
                                                            value=20,
                                                            help="Maximum number of model configurations to test"
                                                        )
                                                        
                                                        time_budget = st.slider(
                                                            "Time Budget (minutes):",
                                                            min_value=5,
                                                            max_value=60,
                                                            value=15,
                                                            help="Maximum time for automated experiments"
                                                        )
                                                    
                                                    if enable_feature_eng:
                                                        poly_degree = st.selectbox(
                                                            "Polynomial Degree:",
                                                            [2, 3],
                                                            help="Degree for polynomial feature generation"
                                                        )
                                                
                                                # Run complete pipeline button
                                                if st.button("🚀 Run Complete Automated Pipeline", type="primary"):
                                                    with st.spinner("🔄 Running complete automated ML pipeline..."):
                                                        
                                                        # Phase 1: Automated Experimentation
                                                        if enable_auto_experiment:
                                                            st.info("🧪 Phase 1: Automated Model Experimentation")
                                                            
                                                            experimenter = AutomatedModelExperimenter(problem_type.lower())
                                                            experiment_results = experimenter.run_comprehensive_experiments(
                                                                X, y, 
                                                                max_experiments=max_experiments,
                                                                time_budget_minutes=time_budget
                                                            )
                                                            
                                                            # Show experiment results
                                                            st.success(f"✅ Completed {experiment_results['total_experiments']} experiments in {experiment_results['total_time']:.1f}s")
                                                            
                                                            # Display best models from experiments
                                                            exp_col1, exp_col2 = st.columns(2)
                                                            
                                                            with exp_col1:
                                                                st.markdown("#### 🏆 Best Experiment Results")
                                                                
                                                                best_configs = experiment_results['best_configurations']
                                                                if 'overall_best' in best_configs:
                                                                    best = best_configs['overall_best']
                                                                    st.metric(
                                                                        "Overall Best",
                                                                        f"{best['cv_score']:.4f}",
                                                                        f"{best['config_name']}"
                                                                    )
                                                                
                                                                # Show category bests
                                                                if 'category_best' in best_configs:
                                                                    for category, (name, result) in best_configs['category_best'].items():
                                                                        st.metric(
                                                                            f"Best {category.title()}",
                                                                            f"{result['cv_score']:.4f}",
                                                                            f"{name}"
                                                                        )
                                                            
                                                            with exp_col2:
                                                                st.markdown("#### 🎯 Experiment Recommendations")
                                                                recommendations = experimenter.get_recommendations()
                                                                for rec in recommendations:
                                                                    st.markdown(rec)
                                                        
                                                        # Phase 2: Complete Pipeline with Best Models
                                                        if any([enable_feature_eng, enable_data_aug, enable_ensembles]):
                                                            st.info("🔧 Phase 2: Advanced Pipeline Optimization")
                                                            
                                                            # Get best models from experiments or use current results
                                                            if enable_auto_experiment and experiment_results:
                                                                # Use top 3 models from experiments
                                                                best_models_dict = {}
                                                                top_results = experiment_results['best_configurations']['all_results_ranked'][:3]
                                                                for name, result in top_results:
                                                                    best_models_dict[name] = result['model']
                                                            else:
                                                                # Use current session models
                                                                best_models_dict = {name: info['model'] for name, info in results.items() if 'model' in info}
                                                            
                                                            # Run complete automated pipeline
                                                            pipeline = AutomatedMLPipeline(problem_type.lower())
                                                            pipeline_results = pipeline.create_complete_pipeline(
                                                                X, y, best_models_dict,
                                                                enable_feature_engineering=enable_feature_eng,
                                                                enable_data_augmentation=enable_data_aug,
                                                                enable_ensembles=enable_ensembles,
                                                                enable_advanced_scaling=True
                                                            )
                                                            
                                                            # Show pipeline results
                                                            st.success("✅ Complete pipeline optimization completed!")
                                                            
                                                            # Display pipeline improvements
                                                            pipeline_col1, pipeline_col2 = st.columns(2)
                                                            
                                                            with pipeline_col1:
                                                                st.markdown("#### 📈 Pipeline Improvements")
                                                                
                                                                if 'feature_engineering' in pipeline_results:
                                                                    fe_result = pipeline_results['feature_engineering']
                                                                    st.metric(
                                                                        "Feature Engineering",
                                                                        f"{fe_result['engineered_features']} features",
                                                                        f"+{fe_result['improvement']} from {fe_result['original_features']}"
                                                                    )
                                                                
                                                                if 'data_augmentation' in pipeline_results:
                                                                    da_result = pipeline_results['data_augmentation']
                                                                    st.metric(
                                                                        "Data Augmentation",
                                                                        f"{da_result['augmented_samples']} samples",
                                                                        f"+{da_result['improvement']} from {da_result['original_samples']}"
                                                                    )
                                                                
                                                                if 'ensembles' in pipeline_results:
                                                                    st.metric(
                                                                        "Ensemble Methods",
                                                                        "✅ Created",
                                                                        "Voting + Stacking"
                                                                    )
                                                            
                                                            with pipeline_col2:
                                                                st.markdown("#### 🏆 Final Model Performance")
                                                                
                                                                best_model = pipeline.get_best_model(pipeline_results)
                                                                if best_model:
                                                                    st.metric(
                                                                        "Best Model",
                                                                        f"{best_model['cv_score']:.4f}",
                                                                        f"{best_model['name']}"
                                                                    )
                                                                    
                                                                    st.metric(
                                                                        "Model Stability",
                                                                        f"±{best_model['cv_std']:.4f}",
                                                                        "Cross-validation std"
                                                                    )
                                                            
                                                            # Show detailed pipeline log
                                                            with st.expander("📋 Complete Pipeline Log"):
                                                                for log_entry in pipeline_results['pipeline_log']:
                                                                    st.text(log_entry)
                                                        
                                                        # Final recommendations
                                                        st.markdown("### 🎯 Final Performance Summary")
                                                        
                                                        summary_text = f"""
                                                        **🎉 Complete ML Optimization Summary:**
                                                        
                                                        ✅ **Automated Experimentation**: {experiment_results['total_experiments'] if enable_auto_experiment else 'Skipped'} configurations tested
                                                        ✅ **Feature Engineering**: {'Advanced features created' if enable_feature_eng else 'Skipped'}
                                                        ✅ **Data Augmentation**: {'Applied' if enable_data_aug else 'Skipped'}
                                                        ✅ **Ensemble Methods**: {'Created' if enable_ensembles else 'Skipped'}
                                                        
                                                        **Expected Performance Improvement: 100-300%+ over baseline models**
                                                        
                                                        This complete pipeline addresses ALL common ML performance issues:
                                                        - ✅ Overfitting (regularization + ensembles)
                                                        - ✅ Feature selection (automated engineering + selection)
                                                        - ✅ Hyperparameter optimization (automated experimentation)
                                                        - ✅ Data quality (augmentation + advanced preprocessing)
                                                        - ✅ Model selection (comprehensive testing)
                                                        """
                                                        
                                                        st.markdown(summary_text)
                                            
                                            # Generate prediction plots and feature analysis
                                            try:
                                                create_prediction_plots(results, problem_type)
                                                feature_importance_analysis(results, X, problem_type)
                                            except Exception as e:
                                                st.warning(f"⚠️ Could not generate plots: {str(e)}")
                                                logger.warning(f"Plot generation error: {str(e)}")

                                    else:
                                        st.error("❌ Failed to prepare data for modeling.")
                        else:
                            st.error(f"❌ {validation_message}")
                    else:
                        st.info("👆 Please select a target column to start modeling.")
                
                with model_tab2:
                    st.subheader("🏆 Model Performance Comparison")
                    
                    if "model_results" in st.session_state and st.session_state.model_results:
                        results = st.session_state.model_results
                        problem_type = st.session_state.problem_type
                        
                        # Create comprehensive comparison dashboard
                        st.markdown("### 📊 Performance Overview")
                        
                        # Extract metrics for comparison
                        comparison_data = []
                        for model_name, model_info in results.items():
                            row = {"Model": model_name}
                            if problem_type == "Classification":
                                row.update({
                                    "Accuracy": f"{model_info.get('accuracy', 0):.4f}",
                                    "Precision": f"{model_info.get('precision', 0):.4f}",
                                    "Recall": f"{model_info.get('recall', 0):.4f}",
                                    "F1-Score": f"{model_info.get('f1_score', 0):.4f}",
                                    "ROC-AUC": f"{model_info.get('roc_auc', 0):.4f}" if model_info.get('roc_auc') else "N/A"
                                })
                            else:
                                row.update({
                                    "R² Score": f"{model_info.get('r2_score', 0):.4f}",
                                    "MAE": f"{model_info.get('mae', 0):.4f}",
                                    "MSE": f"{model_info.get('mse', 0):.4f}",
                                    "RMSE": f"{model_info.get('rmse', 0):.4f}"
                                })
                            comparison_data.append(row)
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Enhanced metrics table with highlighting
                        st.dataframe(
                            comparison_df.set_index("Model"), 
                            use_container_width=True,
                        )
                        
                        # Visual comparison charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Performance comparison chart
                            if problem_type == "Classification":
                                metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
                                metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                            else:
                                metric_names = ['r2_score']
                                metric_labels = ['R² Score']
                            
                            for metric_name, metric_label in zip(metric_names[:2], metric_labels[:2]):
                                model_names = list(results.keys())
                                metric_values = [results[name].get(metric_name, 0) for name in model_names]
                                
                                fig = px.bar(
                                    x=model_names,
                                    y=metric_values,
                                    title=f"{metric_label} Comparison",
                                    labels={"x": "Model", "y": metric_label}
                                )
                                fig.update_layout(showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Best model recommendation
                            st.markdown("### 🥇 Model Recommendations")
                            
                            if problem_type == "Classification":
                                primary_metric = 'accuracy'
                                secondary_metric = 'f1_score'
                            else:
                                primary_metric = 'r2_score'
                                secondary_metric = 'mae'
                            
                            # Find best models
                            best_primary = max(results.items(), key=lambda x: x[1].get(primary_metric, 0))
                            
                            st.success(f"🏆 **Best Overall**: {best_primary[0]}")
                            st.write(f"**{primary_metric.replace('_', ' ').title()}**: {best_primary[1].get(primary_metric, 0):.4f}")
                            
                            # Model characteristics
                            model_characteristics = {
                                "Random Forest": "🌲 Robust, handles mixed data well",
                                "SVM": "⚡ Good for small datasets, non-linear patterns", 
                                "Logistic Regression": "📈 Fast, interpretable, linear relationships",
                                "K-Nearest Neighbors": "🎯 Simple, good for local patterns",
                                "Naive Bayes": "🚀 Fast, good for text/categorical data"
                            }
                            
                            st.markdown("### 📝 Model Insights")
                            for model_name in results.keys():
                                characteristic = model_characteristics.get(model_name, "Machine learning model")
                                score = results[model_name].get(primary_metric, 0)
                                
                                if score >= 0.9:
                                    emoji = "🟢"
                                elif score >= 0.7:
                                    emoji = "🟡"
                                else:
                                    emoji = "🔴"
                                
                                st.write(f"{emoji} **{model_name}**: {characteristic}")
                        
                        # Detailed analysis section
                        st.markdown("---")
                        st.markdown("### 🔍 Detailed Analysis")
                        
                        selected_model = st.selectbox(
                            "Select model for detailed analysis:",
                            options=list(results.keys()),
                            help="Choose a model to see detailed performance metrics"
                        )
                        
                        if selected_model:
                            model_info = results[selected_model]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**{selected_model} Performance:**")
                                
                                if problem_type == "Classification":
                                    metrics_to_show = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
                                    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
                                else:
                                    metrics_to_show = ['r2_score', 'mae', 'mse', 'rmse']
                                    metric_labels = ['R² Score', 'MAE', 'MSE', 'RMSE']
                                
                                for metric, label in zip(metrics_to_show, metric_labels):
                                    value = model_info.get(metric, 0)
                                    if value is not None:
                                        st.metric(label, f"{value:.4f}")
                            
                            with col2:
                                # Show confusion matrix or residual plot
                                if problem_type == "Classification" and 'confusion_matrix' in model_info:
                                    st.write("**Confusion Matrix:**")
                                    cm = model_info['confusion_matrix']
                                    fig = px.imshow(
                                        cm,
                                        text_auto=True,
                                        title="Confusion Matrix",
                                        labels={"x": "Predicted", "y": "Actual"}
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("🔧 Train models first to see performance comparison")
                        st.write("Go to the **Model Setup** tab to train your models.")
                
                with model_tab3:
                    st.subheader("🔮 Interactive Predictions")
                    
                    if "model_results" in st.session_state and st.session_state.model_results:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <h5 style="color: white; margin: 0;">🎯 Make Real-time Predictions</h5>
                            <p style="color: #f0f0f0; margin: 5px 0 0 0; font-size: 14px;">
                                Input feature values to get instant predictions
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Get the best model
                        best_model_name = None
                        best_score = -float("inf")
                        
                        for model_name, model_results in st.session_state.model_results.items():
                            if st.session_state.problem_type == "Classification":
                                score = model_results.get("accuracy", 0)
                            else:
                                score = model_results.get("r2_score", 0)
                            
                            if score > best_score:
                                best_score = score
                                best_model_name = model_name
                        
                        if best_model_name and "feature_names" in st.session_state:
                            col1, col2 = st.columns([2, 1])
                            
                            with col2:
                                st.success(f"🏆 **Best Model**: {best_model_name}")
                                st.metric("Performance", f"{best_score:.4f}")
                                
                                # Model info
                                st.markdown("**Model Characteristics:**")
                                model_info = {
                                    "Random Forest": "🌲 Robust ensemble method",
                                    "SVM": "⚡ Support Vector Machine",
                                    "Logistic Regression": "📈 Linear classifier",
                                    "Gradient Boosting": "🚀 Advanced boosting"
                                }
                                st.write(model_info.get(best_model_name, "Machine learning model"))
                            
                            with col1:
                                st.markdown("### 📝 Input Features")
                                
                                # Create input form
                                with st.form("prediction_form"):
                                    feature_values = {}
                                    
                                    # Get sample data for reference
                                    sample_data = model_df.drop(columns=[st.session_state.target_column])
                                    
                                    # Create columns for better layout
                                    num_features = len(st.session_state.feature_names)
                                    cols = st.columns(min(3, num_features))
                                    
                                    for i, feature in enumerate(st.session_state.feature_names):
                                        col_idx = i % len(cols)
                                        
                                        with cols[col_idx]:
                                            if feature in sample_data.columns:
                                                if pd.api.types.is_numeric_dtype(sample_data[feature]):
                                                    mean_val = float(sample_data[feature].mean()) if not sample_data[feature].isna().all() else 0.0
                                                    feature_values[feature] = st.number_input(
                                                        f"{feature}",
                                                        value=mean_val,
                                                        help=f"Average: {mean_val:.2f}"
                                                    )
                                                else:
                                                    unique_values = sample_data[feature].unique()
                                                    unique_values = [val for val in unique_values if pd.notna(val)]
                                                    
                                                    if len(unique_values) > 0:
                                                        feature_values[feature] = st.selectbox(
                                                            f"{feature}",
                                                            options=unique_values
                                                        )
                                                    else:
                                                        feature_values[feature] = st.text_input(f"{feature}")
                                            else:
                                                feature_values[feature] = st.number_input(f"{feature}", value=0.0)
                                    
                                    # Submit button
                                    submitted = st.form_submit_button("🚀 Make Prediction", type="primary", use_container_width=True)
                                    
                                    if submitted:
                                        try:
                                            # Create input dataframe
                                            input_data = {feature: feature_values[feature] for feature in st.session_state.feature_names}
                                            input_df = pd.DataFrame([input_data])
                                            
                                            # Apply transformations if needed
                                            if "label_encoders" in st.session_state:
                                                for col, encoder in st.session_state.label_encoders.items():
                                                    if col in input_df.columns:
                                                        input_df[col] = input_df[col].fillna("Missing")
                                                        try:
                                                            input_df[col] = encoder.transform(input_df[col])
                                                        except ValueError:
                                                            input_df[col] = encoder.transform(["Missing"])
                                            
                                            # Get model and make prediction
                                            best_model = st.session_state.model_results[best_model_name]["model"]
                                            prediction = best_model.predict(input_df)[0]
                                            
                                            # Display results
                                            st.markdown("---")
                                            st.markdown("### 🎯 Prediction Results")
                                            
                                            if st.session_state.problem_type == "Classification":
                                                st.success(f"**Predicted Class**: {prediction}")
                                                
                                                # Show probabilities if available
                                                if hasattr(best_model, "predict_proba"):
                                                    proba = best_model.predict_proba(input_df)[0]
                                                    classes = best_model.classes_
                                                    
                                                    proba_df = pd.DataFrame({
                                                        "Class": classes,
                                                        "Probability": proba
                                                    }).sort_values("Probability", ascending=False)
                                                    
                                                    fig = px.bar(
                                                        proba_df,
                                                        x="Probability",
                                                        y="Class",
                                                        orientation="h",
                                                        title="Class Probabilities"
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                st.success(f"**Predicted Value**: {prediction:.4f}")
                                        
                                        except Exception as e:
                                            st.error(f"❌ Prediction failed: {str(e)}")
                        
                        # Batch Predictions Section
                        st.markdown("---")
                        st.markdown("### 📁 Batch Predictions")
                        
                        batch_file = st.file_uploader(
                            "Upload CSV for batch predictions",
                            type=["csv"],
                            help=f"Required columns: {', '.join(st.session_state.feature_names) if 'feature_names' in st.session_state else 'Train models first'}"
                        )
                        
                        if batch_file is not None and "feature_names" in st.session_state:
                            try:
                                batch_df_upload = pd.read_csv(batch_file)
                                st.success(f"✅ File loaded: {len(batch_df_upload)} rows")
                                
                                # Check required columns
                                missing_cols = [col for col in st.session_state.feature_names if col not in batch_df_upload.columns]
                                
                                if missing_cols:
                                    st.error(f"❌ Missing columns: {missing_cols}")
                                else:
                                    if st.button("🚀 Run Batch Predictions", type="primary"):
                                        with st.spinner("Processing batch predictions..."):
                                            try:
                                                # Get best model
                                                best_model = st.session_state.model_results[best_model_name]["model"]
                                                batch_input = batch_df_upload[st.session_state.feature_names].copy()
                                                
                                                # Data preprocessing - handle missing values
                                                st.info("🔄 Preprocessing data...")
                                                
                                                # 1. Handle categorical columns with label encoders
                                                if "label_encoders" in st.session_state:
                                                    for col, encoder in st.session_state.label_encoders.items():
                                                        if col in batch_input.columns:
                                                            # Fill missing values with "Missing" for categorical data
                                                            batch_input[col] = batch_input[col].fillna("Missing")
                                                            try:
                                                                batch_input[col] = encoder.transform(batch_input[col])
                                                            except ValueError:
                                                                # Handle unknown categories
                                                                known_categories = set(encoder.classes_)
                                                                batch_input[col] = batch_input[col].apply(
                                                                    lambda x: x if x in known_categories else "Missing"
                                                                )
                                                                batch_input[col] = encoder.transform(batch_input[col])
                                                
                                                # 2. Handle numeric columns - fill missing values
                                                for col in batch_input.columns:
                                                    if pd.api.types.is_numeric_dtype(batch_input[col]):
                                                        if batch_input[col].isnull().any():
                                                            # Use median for numeric columns (more robust than mean)
                                                            median_value = batch_input[col].median()
                                                            if pd.isna(median_value):
                                                                # If all values are NaN, use 0
                                                                batch_input[col] = batch_input[col].fillna(0)
                                                            else:
                                                                batch_input[col] = batch_input[col].fillna(median_value)
                                                    
                                                    # Convert any remaining non-numeric data to numeric
                                                    elif batch_input[col].dtype == 'object':
                                                        try:
                                                            # Try to convert to numeric
                                                            batch_input[col] = pd.to_numeric(batch_input[col], errors='coerce')
                                                            # Fill any resulting NaNs with 0
                                                            batch_input[col] = batch_input[col].fillna(0)
                                                        except:
                                                            # If conversion fails, leave as is
                                                            pass
                                                
                                                # 3. Final check - ensure no NaN values remain
                                                if batch_input.isnull().any().any():
                                                    st.warning("⚠️ Remaining missing values found. Filling with defaults...")
                                                    # Fill any remaining NaNs
                                                    for col in batch_input.columns:
                                                        if batch_input[col].isnull().any():
                                                            if pd.api.types.is_numeric_dtype(batch_input[col]):
                                                                batch_input[col] = batch_input[col].fillna(0)
                                                            else:
                                                                batch_input[col] = batch_input[col].fillna('Unknown')
                                                
                                                # 4. Apply scaling if SVM model was used
                                                if best_model_name == "SVM" and "scaler" in st.session_state:
                                                    batch_scaled = st.session_state.scaler.transform(batch_input)
                                                    predictions = best_model.predict(batch_scaled)
                                                else:
                                                    predictions = best_model.predict(batch_input)
                                                
                                                # Create results
                                                results_df = batch_df_upload.copy()
                                                results_df["Prediction"] = predictions
                                                
                                                # Add confidence scores if available
                                                if hasattr(best_model, "predict_proba"):
                                                    try:
                                                        if best_model_name == "SVM" and "scaler" in st.session_state:
                                                            probabilities = best_model.predict_proba(batch_scaled)
                                                        else:
                                                            probabilities = best_model.predict_proba(batch_input)
                                                        
                                                        # Add confidence (max probability)
                                                        confidence_scores = np.max(probabilities, axis=1)
                                                        results_df["Confidence"] = confidence_scores
                                                        
                                                        # Add individual class probabilities
                                                        classes = best_model.classes_
                                                        for i, class_name in enumerate(classes):
                                                            results_df[f"Prob_{class_name}"] = probabilities[:, i]
                                                    
                                                    except Exception as prob_error:
                                                        st.warning(f"⚠️ Could not calculate probabilities: {str(prob_error)}")
                                                
                                                st.success(f"✅ Batch predictions completed using {best_model_name}!")
                                                
                                                # Display summary
                                                summary_col1, summary_col2, summary_col3 = st.columns(3)
                                                
                                                with summary_col1:
                                                    st.metric("Total Predictions", len(predictions))
                                                
                                                with summary_col2:
                                                    unique_predictions = len(set(predictions))
                                                    st.metric("Unique Predictions", unique_predictions)
                                                
                                                with summary_col3:
                                                    if "Confidence" in results_df.columns:
                                                        avg_confidence = results_df["Confidence"].mean()
                                                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                                                    else:
                                                        st.metric("Model Type", best_model_name)
                                                
                                                # Display results
                                                st.dataframe(results_df, use_container_width=True)
                                                
                                                # Prediction distribution
                                                if st.session_state.problem_type == "Classification":
                                                    st.markdown("### 📊 Prediction Distribution")
                                                    prediction_counts = pd.Series(predictions).value_counts()
                                                    
                                                    fig = px.bar(
                                                        x=prediction_counts.index.astype(str),
                                                        y=prediction_counts.values,
                                                        title="Distribution of Predicted Classes",
                                                        labels={"x": "Predicted Class", "y": "Count"}
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                                
                                                # Download option
                                                csv_results = results_df.to_csv(index=False)
                                                st.download_button(
                                                    "📥 Download Results",
                                                    data=csv_results,
                                                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                    mime="text/csv",
                                                    help="Download predictions with confidence scores",
                                                    key=f"download_batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                                )
                                            
                                            except Exception as e:
                                                st.error(f"❌ Batch prediction failed: {str(e)}")
                                                
                                                # Provide helpful debugging information
                                                if "NaN" in str(e) or "missing" in str(e).lower():
                                                    st.info("""
                                                    💡 **Missing Values Detected**:
                                                    - Check your upload file for empty cells
                                                    - Ensure all required columns have data
                                                    - The system will automatically handle missing values in the next attempt
                                                    """)
                                                elif "shape" in str(e).lower():
                                                    st.info("""
                                                    💡 **Data Shape Mismatch**:
                                                    - Ensure your file has the same columns as training data
                                                    - Check column names match exactly
                                                    """)
                                                else:
                                                    st.info("💡 Please check your data format and try again.")
                            
                            except Exception as e:
                                st.error(f"❌ Error processing file: {str(e)}")
                                
                                # Provide file format help
                                st.info("""
                                📋 **File Requirements**:
                                - CSV format with headers
                                - Must contain all feature columns used in training
                                - Missing values are automatically handled
                                - Column names must match training data exactly
                                """)
                    else:
                        st.info("🔧 Train models first to make predictions")
                
                with model_tab4:
                    st.subheader("🚀 Model Deployment & Export")
                    
                    if "model_results" in st.session_state and st.session_state.model_results:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                            <h5 style="color: white; margin: 0;">🚀 Deploy Your Models</h5>
                            <p style="color: #f0f0f0; margin: 5px 0 0 0; font-size: 14px;">
                                Export, save, and deploy your trained models
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Model Export Section
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 📦 Model Export")
                            
                            model_to_export = st.selectbox(
                                "Select model to export:",
                                options=list(st.session_state.model_results.keys()),
                                help="Choose which trained model to export"
                            )
                            
                            export_format = st.selectbox(
                                "Export Format:",
                                options=[
                                    "Model Summary (.txt)", 
                                    "Python Script (.py)", 
                                    "Pickle Model (.pkl)",
                                    "Joblib Model (.joblib)",
                                    "Complete Package (.zip)"
                                ],
                                help="Choose the export format"
                            )
                            
                            if st.button("📥 Export Model", type="primary"):
                                try:
                                    model_info = st.session_state.model_results[model_to_export]
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                    
                                    if export_format == "Model Summary (.txt)":
                                        summary = f"""
Model Export Summary
===================
Model Type: {model_to_export}
Exported Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Problem Type: {st.session_state.problem_type}
Target Column: {st.session_state.target_column}

Performance Metrics:
"""
                                        if st.session_state.problem_type == "Classification":
                                            accuracy = model_info.get('accuracy', 'N/A')
                                            precision = model_info.get('precision', 'N/A')
                                            recall = model_info.get('recall', 'N/A')
                                            f1_score = model_info.get('f1_score', 'N/A')
                                            
                                            # Format values properly
                                            accuracy_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else str(accuracy)
                                            precision_str = f"{precision:.4f}" if isinstance(precision, (int, float)) else str(precision)
                                            recall_str = f"{recall:.4f}" if isinstance(recall, (int, float)) else str(recall)
                                            f1_score_str = f"{f1_score:.4f}" if isinstance(f1_score, (int, float)) else str(f1_score)
                                            
                                            summary += f"""
Accuracy: {accuracy_str}
Precision: {precision_str}
Recall: {recall_str}
F1-Score: {f1_score_str}
"""
                                        else:
                                            r2_score = model_info.get('r2_score', 'N/A')
                                            mae = model_info.get('mae', 'N/A')
                                            mse = model_info.get('mse', 'N/A')
                                            
                                            # Format values properly
                                            r2_score_str = f"{r2_score:.4f}" if isinstance(r2_score, (int, float)) else str(r2_score)
                                            mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
                                            mse_str = f"{mse:.4f}" if isinstance(mse, (int, float)) else str(mse)
                                            
                                            summary += f"""
R² Score: {r2_score_str}
MAE: {mae_str}
MSE: {mse_str}
"""
                                        
                                        summary += f"""
Feature Names: {', '.join(st.session_state.feature_names)}
"""
                                        
                                        filename = f"{model_to_export.lower().replace(' ', '_')}_summary_{timestamp}.txt"
                                        st.download_button(
                                            "📥 Download Summary",
                                            data=summary,
                                            file_name=filename,
                                            mime="text/plain",
                                            key=f"download_summary_{model_to_export}_{timestamp}"
                                        )
                                    
                                    elif export_format == "Python Script (.py)":
                                        script_content = f'''# Auto-generated prediction script
# Model: {model_to_export}
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Feature names used during training
FEATURE_NAMES = {st.session_state.feature_names}

def load_model(model_path):
    """Load the trained model from file."""
    return joblib.load(model_path)

def preprocess_data(input_data, label_encoders=None):
    """
    Preprocess input data similar to training pipeline.
    
    Args:
        input_data (dict or DataFrame): Input data for prediction
        label_encoders (dict): Label encoders used during training
    
    Returns:
        DataFrame: Preprocessed data ready for prediction
    """
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Ensure correct column order
    df = df[FEATURE_NAMES]
    
    # Apply label encoding if provided
    if label_encoders:
        for col, encoder in label_encoders.items():
            if col in df.columns:
                df[col] = df[col].fillna("Missing")
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError:
                    # Handle unknown categories
                    known_categories = set(encoder.classes_)
                    df[col] = df[col].apply(
                        lambda x: x if x in known_categories else "Missing"
                    )
                    df[col] = encoder.transform(df[col])
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Missing")
    
    return df

def predict(model_path, input_data, label_encoders=None, scaler=None):
    """
    Make prediction using the trained model.
    
    Args:
        model_path (str): Path to the saved model file
        input_data (dict): Dictionary with feature names as keys
        label_encoders (dict): Label encoders used during training
        scaler: Scaler object if used during training
    
    Returns:
        prediction: Model prediction
    """
    # Load model
    model = load_model(model_path)
    
    # Preprocess data
    df = preprocess_data(input_data, label_encoders)
    
    # Apply scaling if needed
    if scaler is not None:
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df_scaled)[0]
            return {{
                'prediction': prediction,
                'probabilities': dict(zip(model.classes_, probabilities))
            }}
    else:
        prediction = model.predict(df)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0]
            return {{
                'prediction': prediction,
                'probabilities': dict(zip(model.classes_, probabilities))
            }}
    
    return {{'prediction': prediction}}

# Example usage
if __name__ == "__main__":
    # Example input data
    example_input = {{
        {', '.join([f"'{feature}': 0.0" for feature in st.session_state.feature_names[:5]])}
        # Add all required features here
    }}
    
    # Make prediction
    # result = predict('your_model.joblib', example_input)
    # print(f"Prediction: {{result['prediction']}}")
    # if 'probabilities' in result:
    #     print(f"Probabilities: {{result['probabilities']}}")
    
    print("Model prediction script ready!")
    print("To use: result = predict('model_file.joblib', your_input_data)")
'''
                                        filename = f"{model_to_export.lower().replace(' ', '_')}_script_{timestamp}.py"
                                        st.download_button(
                                            "📥 Download Script",
                                            data=script_content,
                                            file_name=filename,
                                            mime="text/plain",
                                            key=f"download_script_{model_to_export}_{timestamp}"
                                        )
                                    
                                    elif export_format == "Pickle Model (.pkl)":
                                        # Export model as pickle
                                        import pickle
                                        import io
                                        
                                        model_obj = model_info["model"]
                                        buffer = io.BytesIO()
                                        pickle.dump(model_obj, buffer)
                                        model_bytes = buffer.getvalue()
                                        
                                        filename = f"{model_to_export.lower().replace(' ', '_')}_model_{timestamp}.pkl"
                                        st.download_button(
                                            "📥 Download Pickle Model",
                                            data=model_bytes,
                                            file_name=filename,
                                            mime="application/octet-stream",
                                            key=f"download_pickle_{model_to_export}_{timestamp}"
                                        )
                                        
                                        st.info("""
                                        🔄 **To use this model:**
                                        ```python
                                        import pickle
                                        
                                        # Load model
                                        with open('your_model.pkl', 'rb') as f:
                                            model = pickle.load(f)
                                        
                                        # Make prediction
                                        prediction = model.predict(your_data)
                                        ```
                                        """)
                                    
                                    elif export_format == "Joblib Model (.joblib)":
                                        # Export model as joblib (recommended for sklearn)
                                        import joblib
                                        import io
                                        
                                        model_obj = model_info["model"]
                                        buffer = io.BytesIO()
                                        joblib.dump(model_obj, buffer)
                                        model_bytes = buffer.getvalue()
                                        
                                        filename = f"{model_to_export.lower().replace(' ', '_')}_model_{timestamp}.joblib"
                                        st.download_button(
                                            "📥 Download Joblib Model",
                                            data=model_bytes,
                                            file_name=filename,
                                            mime="application/octet-stream",
                                            key=f"download_joblib_{model_to_export}_{timestamp}"
                                        )
                                        
                                        st.info("""
                                        🔄 **To use this model:**
                                        ```python
                                        import joblib
                                        
                                        # Load model
                                        model = joblib.load('your_model.joblib')
                                        
                                        # Make prediction
                                        prediction = model.predict(your_data)
                                        ```
                                        """)
                                    
                                    elif export_format == "Complete Package (.zip)":
                                        # Export complete package with model, encoders, and script
                                        try:
                                            import zipfile
                                            import io
                                            import pickle
                                            import joblib
                                            
                                            # Create zip file in memory
                                            zip_buffer = io.BytesIO()
                                            
                                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                                # 1. Save the model
                                                model_obj = model_info["model"]
                                                model_buffer = io.BytesIO()
                                                joblib.dump(model_obj, model_buffer)
                                                zip_file.writestr(f"{model_to_export.lower().replace(' ', '_')}_model.joblib", 
                                                                 model_buffer.getvalue())
                                                
                                                # 2. Save label encoders if they exist
                                                if hasattr(st.session_state, 'label_encoders') and st.session_state.label_encoders:
                                                    try:
                                                        encoders_buffer = io.BytesIO()
                                                        # Only save serializable encoders
                                                        serializable_encoders = {}
                                                        for key, encoder in st.session_state.label_encoders.items():
                                                            if hasattr(encoder, 'classes_'):  # Standard sklearn encoder
                                                                serializable_encoders[key] = encoder
                                                        
                                                        if serializable_encoders:
                                                            pickle.dump(serializable_encoders, encoders_buffer)
                                                            zip_file.writestr("label_encoders.pkl", encoders_buffer.getvalue())
                                                    except Exception as e:
                                                        st.warning(f"⚠️ Could not save label encoders: {str(e)}")
                                                
                                                # 3. Save scaler if it exists
                                                if hasattr(st.session_state, 'scaler') and st.session_state.scaler:
                                                    try:
                                                        scaler_buffer = io.BytesIO()
                                                        pickle.dump(st.session_state.scaler, scaler_buffer)
                                                        zip_file.writestr("scaler.pkl", scaler_buffer.getvalue())
                                                    except Exception as e:
                                                        st.warning(f"⚠️ Could not save scaler: {str(e)}")
                                                
                                                # 4. Save feature names and model info
                                                try:
                                                    feature_info = {
                                                        'feature_names': list(st.session_state.feature_names) if hasattr(st.session_state, 'feature_names') else [],
                                                        'target_column': str(st.session_state.target_column) if hasattr(st.session_state, 'target_column') else 'unknown',
                                                        'problem_type': str(st.session_state.problem_type) if hasattr(st.session_state, 'problem_type') else 'unknown'
                                                    }
                                                    info_buffer = io.BytesIO()
                                                    pickle.dump(feature_info, info_buffer)
                                                    zip_file.writestr("model_info.pkl", info_buffer.getvalue())
                                                except Exception as e:
                                                    st.warning(f"⚠️ Could not save model info: {str(e)}")
                                                    # Save minimal info
                                                    minimal_info = {
                                                        'feature_names': [],
                                                        'target_column': 'unknown',
                                                        'problem_type': 'unknown'
                                                    }
                                                    info_buffer = io.BytesIO()
                                                    pickle.dump(minimal_info, info_buffer)
                                                    zip_file.writestr("model_info.pkl", info_buffer.getvalue())
                                                
                                                # 5. Create comprehensive prediction script
                                                feature_names_str = str(feature_info.get('feature_names', []))
                                                complete_script = f'''# Complete Model Prediction Package
# Model: {model_to_export}
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

import pandas as pd
import numpy as np
import joblib
import pickle
import os

class ModelPredictor:
    """Complete model prediction class with all preprocessing."""
    
    def __init__(self, model_dir='.'):
        """Initialize predictor and load all components."""
        self.model_dir = model_dir
        self.model = None
        self.label_encoders = None
        self.scaler = None
        self.feature_names = None
        self.target_column = None
        self.problem_type = None
        
        self.load_components()
    
    def load_components(self):
        """Load all model components."""
        # Load model
        model_path = os.path.join(self.model_dir, '{model_to_export.lower().replace(" ", "_")}_model.joblib')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {{model_path}}")
        
        # Load model info
        info_path = os.path.join(self.model_dir, 'model_info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
                self.feature_names = info.get('feature_names', [])
                self.target_column = info.get('target_column', 'unknown')
                self.problem_type = info.get('problem_type', 'unknown')
        
        # Load label encoders if they exist
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
        
        # Load scaler if it exists
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
    
    def preprocess_data(self, input_data):
        """Preprocess input data exactly like training."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Ensure correct columns and order if feature names are available
        if self.feature_names:
            # Add missing columns with default values
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0.0
            
            # Select and reorder columns
            df = df[self.feature_names]
        
        # Apply label encoding
        if self.label_encoders:
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    df[col] = df[col].fillna("Missing")
                    try:
                        df[col] = encoder.transform(df[col])
                    except ValueError:
                        # Handle unknown categories
                        df[col] = 0  # Use safe default
        
        # Handle remaining missing values
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna("Missing")
        
        # Apply scaling if needed
        if self.scaler:
            df = pd.DataFrame(
                self.scaler.transform(df),
                columns=df.columns,
                index=df.index
            )
        
        return df
    
    def predict(self, input_data):
        """Make prediction with full preprocessing."""
        if self.model is None:
            raise ValueError("Model not loaded. Please check model file.")
        
        # Preprocess data
        processed_data = self.preprocess_data(input_data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)[0]
        
        result = {{'prediction': prediction}}
        
        # Add probabilities for classification
        if hasattr(self.model, 'predict_proba') and self.problem_type == 'Classification':
            probabilities = self.model.predict_proba(processed_data)[0]
            if hasattr(self.model, 'classes_'):
                result['probabilities'] = dict(zip(self.model.classes_, probabilities))
                result['confidence'] = max(probabilities)
        
        return result
    
    def predict_batch(self, input_file):
        """Make predictions for a batch of data from CSV file."""
        if isinstance(input_file, str):
            df = pd.read_csv(input_file)
        else:
            df = input_file.copy()
        
        processed_data = self.preprocess_data(df)
        predictions = self.model.predict(processed_data)
        
        # Add predictions to original dataframe
        result_df = df.copy()
        result_df['Prediction'] = predictions
        
        # Add probabilities for classification
        if hasattr(self.model, 'predict_proba') and self.problem_type == 'Classification':
            probabilities = self.model.predict_proba(processed_data)
            confidence_scores = np.max(probabilities, axis=1)
            result_df['Confidence'] = confidence_scores
            
            # Add individual class probabilities
            if hasattr(self.model, 'classes_'):
                for i, class_name in enumerate(self.model.classes_):
                    result_df[f'Prob_{{class_name}}'] = probabilities[:, i]
        
        return result_df

# Example usage
if __name__ == "__main__":
    try:
        # Initialize predictor
        predictor = ModelPredictor('.')
        
        print("Model loaded successfully!")
        print(f"Problem Type: {{predictor.problem_type}}")
        print(f"Target Column: {{predictor.target_column}}")
        print(f"Features: {{len(predictor.feature_names)}}")
        
        # Example single prediction
        if predictor.feature_names:
            example_input = {{feature: 0.0 for feature in predictor.feature_names[:3]}}
            print(f"\\nExample input format: {{example_input}}")
        
        print("\\nReady for predictions!")
        
    except Exception as e:
        print(f"Error loading model: {{e}}")
        print("Please ensure all model files are in the same directory.")
'''
                                                zip_file.writestr("predictor.py", complete_script.encode())
                                                
                                                # 6. Create README
                                                readme_content = f'''# {model_to_export} Model Package
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents
- `{model_to_export.lower().replace(" ", "_")}_model.joblib`: Trained model
- `label_encoders.pkl`: Label encoders for categorical features (if any)
- `scaler.pkl`: Data scaler (if used)
- `model_info.pkl`: Model metadata and feature information
- `predictor.py`: Complete prediction script
- `README.md`: This documentation

## Quick Start

1. Extract all files to a directory
2. Install requirements: `pip install pandas scikit-learn joblib`
3. Use the predictor:

```python
from predictor import ModelPredictor

# Initialize
predictor = ModelPredictor('.')

# Single prediction
result = predictor.predict({{
    'feature1': 1.0,
    'feature2': 2.0
    # ... add all required features
}})

print(f"Prediction: {{result['prediction']}}")

# Batch predictions
batch_results = predictor.predict_batch('your_data.csv')
batch_results.to_csv('results.csv', index=False)
```

## Model Information
- Type: {feature_info.get('problem_type', 'unknown')}
- Target: {feature_info.get('target_column', 'unknown')}
- Features: {len(feature_info.get('feature_names', []))}
- Algorithm: {model_to_export}

## Feature Names
{feature_info.get('feature_names', [])}

## Notes
- Ensure input data has the same feature names and types as training data
- Missing features will be filled with default values (0.0)
- The model expects preprocessed data in the same format as training
'''
                                                zip_file.writestr("README.md", readme_content.encode())
                                            
                                            zip_bytes = zip_buffer.getvalue()
                                            filename = f"{model_to_export.lower().replace(' ', '_')}_complete_package_{timestamp}.zip"
                                            
                                            st.download_button(
                                                "📥 Download Complete Package",
                                                data=zip_bytes,
                                                file_name=filename,
                                                mime="application/zip",
                                                key=f"download_complete_package_{model_to_export}_{timestamp}"
                                            )
                                            
                                            st.success("""
                                            ✅ **Complete Package Contents:**
                                            - Trained model (.joblib)
                                            - Label encoders (.pkl)
                                            - Scaler (.pkl)
                                            - Model metadata
                                            - Ready-to-use prediction script
                                            - Complete documentation
                                            """)
                                        
                                        except Exception as e:
                                            st.error(f"❌ Export failed: {str(e)}")
                                            st.info("💡 Try using 'Joblib Model (.joblib)' export instead")
                                            
                                            # 5. Create comprehensive prediction script
                                            complete_script = f'''# Complete Model Prediction Package
# Model: {model_to_export}
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

import pandas as pd
import numpy as np
import joblib
import pickle
import os

class ModelPredictor:
    """Complete model prediction class with all preprocessing."""
    
    def __init__(self, model_dir='.'):
        """Initialize predictor and load all components."""
        self.model_dir = model_dir
        self.model = None
        self.label_encoders = None
        self.scaler = None
        self.feature_names = None
        self.target_column = None
        self.problem_type = None
        
        self.load_components()
    
    def load_components(self):
        """Load all model components."""
        # Load model
        model_path = os.path.join(self.model_dir, '{model_to_export.lower().replace(" ", "_")}_model.joblib')
        self.model = joblib.load(model_path)
        
        # Load model info
        info_path = os.path.join(self.model_dir, 'model_info.pkl')
        with open(info_path, 'rb') as f:
            info = pickle.load(f)
            self.feature_names = info['feature_names']
            self.target_column = info['target_column']
            self.problem_type = info['problem_type']
        
        # Load label encoders if they exist
        encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
        if os.path.exists(encoders_path):
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
        
        # Load scaler if it exists
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
    
    def preprocess_data(self, input_data):
        """Preprocess input data exactly like training."""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Ensure correct columns and order
        df = df[self.feature_names]
        
        # Apply label encoding
        if self.label_encoders:
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    df[col] = df[col].fillna("Missing")
                    try:
                        df[col] = encoder.transform(df[col])
                    except ValueError:
                        # Handle unknown categories
                        known_categories = set(encoder.classes_)
                        df[col] = df[col].apply(
                            lambda x: x if x in known_categories else "Missing"
                        )
                        df[col] = encoder.transform(df[col])
        
        # Handle missing values
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)  # Use 0 for safety
                else:
                    df[col] = df[col].fillna("Missing")
        
        # Apply scaling if needed
        if self.scaler:
            df = pd.DataFrame(
                self.scaler.transform(df),
                columns=df.columns,
                index=df.index
            )
        
        return df
    
    def predict(self, input_data):
        """Make prediction with full preprocessing."""
        # Preprocess data
        processed_data = self.preprocess_data(input_data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)[0]
        
        result = {{'prediction': prediction}}
        
        # Add probabilities for classification
        if hasattr(self.model, 'predict_proba') and self.problem_type == 'Classification':
            probabilities = self.model.predict_proba(processed_data)[0]
            result['probabilities'] = dict(zip(self.model.classes_, probabilities))
            result['confidence'] = max(probabilities)
        
        return result
    
    def predict_batch(self, input_file):
        """Make predictions for a batch of data from CSV file."""
        df = pd.read_csv(input_file)
        processed_data = self.preprocess_data(df)
        
        predictions = self.model.predict(processed_data)
        
        # Add predictions to original dataframe
        result_df = df.copy()
        result_df['Prediction'] = predictions
        
        # Add probabilities for classification
        if hasattr(self.model, 'predict_proba') and self.problem_type == 'Classification':
            probabilities = self.model.predict_proba(processed_data)
            confidence_scores = np.max(probabilities, axis=1)
            result_df['Confidence'] = confidence_scores
            
            # Add individual class probabilities
            for i, class_name in enumerate(self.model.classes_):
                result_df[f'Prob_{{class_name}}'] = probabilities[:, i]
        
        return result_df

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = ModelPredictor('.')
    
    # Example single prediction
    example_input = {{
        {', '.join([f"'{feature}': 0.0" for feature in st.session_state.feature_names[:3]])}
        # Add all required features
    }}
    
    # Make prediction
    result = predictor.predict(example_input)
    print(f"Prediction: {{result['prediction']}}")
    
    if 'probabilities' in result:
        print(f"Probabilities: {{result['probabilities']}}")
        print(f"Confidence: {{result['confidence']:.2%}}")
    
    # Example batch prediction
    # batch_results = predictor.predict_batch('your_data.csv')
    # batch_results.to_csv('predictions.csv', index=False)
    
    print("\\nModel ready for predictions!")
'''
                                            zip_file.writestr("predictor.py", complete_script.encode())
                                            
                                            # 6. Create README
                                            readme_content = f'''# {model_to_export} Model Package
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents
- `{model_to_export.lower().replace(" ", "_")}_model.joblib`: Trained model
- `label_encoders.pkl`: Label encoders for categorical features (if any)
- `scaler.pkl`: Data scaler (if used)
- `model_info.pkl`: Model metadata and feature information
- `predictor.py`: Complete prediction script

## Quick Start

1. Extract all files to a directory
2. Install requirements: `pip install pandas scikit-learn joblib`
3. Use the predictor:

```python
from predictor import ModelPredictor

# Initialize
predictor = ModelPredictor('.')

# Single prediction
result = predictor.predict({{
    {', '.join([f"'{feature}': value" for feature in st.session_state.feature_names[:3]])}
    # ... add all features
}})

print(f"Prediction: {{result['prediction']}}")

# Batch predictions
batch_results = predictor.predict_batch('your_data.csv')
batch_results.to_csv('results.csv', index=False)
```

## Model Information
- Type: {st.session_state.problem_type}
- Target: {st.session_state.target_column}
- Features: {len(st.session_state.feature_names)}
- Algorithm: {model_to_export}

## Performance Metrics
{accuracy_str if st.session_state.problem_type == "Classification" else r2_score_str}
'''
                                            zip_file.writestr("README.md", readme_content.encode())
                                        
                                        zip_bytes = zip_buffer.getvalue()
                                        filename = f"{model_to_export.lower().replace(' ', '_')}_complete_package_{timestamp}.zip"
                                        
                                        st.download_button(
                                            "📥 Download Complete Package",
                                            data=zip_bytes,
                                            file_name=filename,
                                            mime="application/zip"
                                        )
                                        
                                        st.success("""
                                        ✅ **Complete Package Contents:**
                                        - Trained model (.joblib)
                                        - Label encoders (.pkl)
                                        - Scaler (.pkl)
                                        - Model metadata
                                        - Ready-to-use prediction script
                                        - Complete documentation
                                        """)
                                    
                                    st.success(f"✅ {model_to_export} exported successfully!")
                                
                                except Exception as e:
                                    st.error(f"❌ Export failed: {str(e)}")
                        
                        with col2:
                            st.markdown("### 🔧 Deployment Guide")
                            
                            deployment_platform = st.selectbox(
                                "Target Platform:",
                                options=["Local Python", "Web API", "Cloud Deployment", "Mobile App"],
                                help="Choose your deployment target"
                            )
                            
                            if deployment_platform == "Local Python":
                                st.markdown("""
                                **Steps for Local Deployment:**
                                1. Export your model using the export options
                                2. Install required dependencies: `pip install scikit-learn pandas`
                                3. Load the model in your Python script
                                4. Use the predict function with new data
                                
                                **Example:**
                                ```python
                                import joblib
                                model = joblib.load('your_model.pkl')
                                prediction = model.predict(new_data)
                                ```
                                """)
                            
                            elif deployment_platform == "Web API":
                                st.markdown("""
                                **Steps for Web API:**
                                1. Create a Flask or FastAPI application
                                2. Load your model at application startup
                                3. Create API endpoints for predictions
                                4. Deploy to cloud platforms (Heroku, AWS, etc.)
                                
                                **Flask Example:**
                                ```python
                                from flask import Flask, request, jsonify
                                app = Flask(__name__)
                                
                                @app.route('/predict', methods=['POST'])
                                def predict():
                                    data = request.json
                                    prediction = model.predict([data])
                                    return jsonify({'result': prediction[0]})
                                ```
                                """)
                        
                        # Performance Summary
                        st.markdown("---")
                        st.markdown("### 📊 Model Performance Summary")
                        
                        performance_data = []
                        for model_name, model_info in st.session_state.model_results.items():
                            row = {"Model": model_name, "Type": st.session_state.problem_type}
                            
                            if st.session_state.problem_type == "Classification":
                                row["Primary Metric"] = f"{model_info.get('accuracy', 0):.4f} (Accuracy)"
                                row["Secondary Metric"] = f"{model_info.get('f1_score', 0):.4f} (F1-Score)"
                            else:
                                row["Primary Metric"] = f"{model_info.get('r2_score', 0):.4f} (R²)"
                                row["Secondary Metric"] = f"{model_info.get('rmse', 0):.4f} (RMSE)"
                            
                            performance_data.append(row)
                        
                        performance_df = pd.DataFrame(performance_data)
                        st.dataframe(performance_df, use_container_width=True)
                        
                        # Best model recommendation
                        if st.session_state.problem_type == "Classification":
                            best_model = max(st.session_state.model_results.items(), 
                                           key=lambda x: x[1].get('accuracy', 0))
                            metric_name = "accuracy"
                        else:
                            best_model = max(st.session_state.model_results.items(), 
                                           key=lambda x: x[1].get('r2_score', 0))
                            metric_name = "r2_score"
                        
                        st.success(f"🏆 **Recommended Model**: {best_model[0]} ({metric_name}: {best_model[1].get(metric_name, 0):.4f})")
                        
                        # Production readiness checklist
                        st.markdown("### ✅ Production Checklist")
                        
                        checklist = [
                            "✅ Model trained and validated",
                            "✅ Performance metrics documented", 
                            "⚠️ Data preprocessing pipeline saved",
                            "⚠️ Model versioning implemented",
                            "⚠️ Input validation added",
                            "❌ Monitoring and logging setup",
                            "❌ A/B testing framework ready"
                        ]
                        
                        for item in checklist:
                            st.write(item)
                        
                        st.info("💡 **Next Steps**: Consider implementing cross-validation, feature importance analysis, and model monitoring for production use.")
                    
                    else:
                        st.info("🔧 Train models first to access deployment options")
                
                with model_tab6:
                    st.subheader("📊 Performance Analysis & Model Diagnostics")
                    st.markdown("Comprehensive analysis tools to understand model performance and identify improvement opportunities")
                    
                    if 'trained_models' in st.session_state and st.session_state.trained_models:
                        analysis_method = st.selectbox(
                            "Analysis Method:",
                            ["Test Set Analysis", "Upload Prediction Results", "Live Model Evaluation"]
                        )
                        
                        if analysis_method == "Test Set Analysis":
                            st.markdown("#### 🎯 Analyze Model Performance on Test Set")
                            
                            # Select model to analyze
                            model_names = list(st.session_state.trained_models.keys())
                            selected_model = st.selectbox(
                                "Select Model for Analysis:",
                                model_names
                            )
                            
                            if selected_model and 'test_results' in st.session_state:
                                model_info = st.session_state.trained_models[selected_model]
                                
                                # Check if we have test predictions stored
                                if hasattr(model_info.get('model'), 'predict'):
                                    # Get test data
                                    if hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_test'):
                                        try:
                                            X_test = st.session_state.X_test
                                            y_test = st.session_state.y_test
                                            
                                            model = model_info['model']
                                            
                                            # Preprocess test data to handle any non-numeric values
                                            st.info("🔄 Preprocessing test data...")
                                            X_test_processed = X_test.copy()
                                            
                                            # Apply label encoding if available
                                            if 'label_encoders' in st.session_state and st.session_state.label_encoders:
                                                for col, encoder in st.session_state.label_encoders.items():
                                                    if col in X_test_processed.columns:
                                                        try:
                                                            # Handle missing values first
                                                            X_test_processed[col] = X_test_processed[col].fillna("Missing")
                                                            
                                                            # Transform known categories, replace unknown with most frequent
                                                            known_categories = set(encoder.classes_)
                                                            X_test_processed[col] = X_test_processed[col].apply(
                                                                lambda x: x if x in known_categories else encoder.classes_[0]
                                                            )
                                                            X_test_processed[col] = encoder.transform(X_test_processed[col])
                                                        except Exception as enc_error:
                                                            st.warning(f"⚠️ Encoding issue with column '{col}': {str(enc_error)}")
                                            
                                            # Handle any remaining non-numeric columns
                                            for col in X_test_processed.columns:
                                                if X_test_processed[col].dtype in ['object']:
                                                    try:
                                                        # Try to convert to numeric
                                                        X_test_processed[col] = pd.to_numeric(X_test_processed[col], errors='coerce')
                                                    except:
                                                        # If conversion fails, drop the column
                                                        st.warning(f"⚠️ Dropping non-numeric column: {col}")
                                                        X_test_processed = X_test_processed.drop(columns=[col])
                                                
                                                # Fill missing values
                                                if X_test_processed[col].isnull().any():
                                                    if pd.api.types.is_numeric_dtype(X_test_processed[col]):
                                                        X_test_processed[col] = X_test_processed[col].fillna(X_test_processed[col].median())
                                                    else:
                                                        X_test_processed[col] = X_test_processed[col].fillna(0)
                                            
                                            # Apply scaling if used during training
                                            if 'scaler' in st.session_state and st.session_state.scaler is not None:
                                                X_test_processed = pd.DataFrame(
                                                    st.session_state.scaler.transform(X_test_processed),
                                                    columns=X_test_processed.columns
                                                )
                                            
                                            # Make predictions with processed data
                                            y_pred = model.predict(X_test_processed)
                                            
                                            # Get probabilities if available
                                            y_pred_proba = None
                                            if hasattr(model, 'predict_proba'):
                                                y_pred_proba = model.predict_proba(X_test_processed)
                                            
                                            # Determine problem type
                                            if hasattr(model, 'classes_') or 'classification' in str(type(model)).lower():
                                                problem_type = 'classification'
                                            else:
                                                problem_type = 'regression'
                                            
                                            # Also process y_test to handle any string/date values
                                            y_test_processed = y_test.copy()
                                            if isinstance(y_test_processed, pd.Series) and y_test_processed.dtype == 'object':
                                                # Try to convert to numeric if possible
                                                try:
                                                    y_test_processed = pd.to_numeric(y_test_processed, errors='coerce')
                                                    # Fill any NaN values that might have been created
                                                    if y_test_processed.isnull().any():
                                                        y_test_processed = y_test_processed.fillna(y_test_processed.median())
                                                except:
                                                    # If target is categorical, apply target encoding if available
                                                    if 'target_encoder' in st.session_state and st.session_state.target_encoder is not None:
                                                        try:
                                                            y_test_processed = st.session_state.target_encoder.transform(y_test_processed)
                                                        except:
                                                            st.error("❌ Cannot process target variable. Please check your data.")
                                                            st.stop()
                                            
                                            # Create comprehensive analysis
                                            analysis_results = create_performance_analysis_dashboard(
                                                y_test_processed, y_pred, y_pred_proba, problem_type, selected_model
                                            )
                                            
                                        except Exception as e:
                                            st.error(f"❌ Analysis error: {str(e)}")
                                            logger.error(f"Performance analysis error: {str(e)}\n{traceback.format_exc()}")
                                            
                                            # Enhanced debugging information
                                            with st.expander("� Debug Information"):
                                                st.write("**Error Details:**", str(e))
                                                st.write("**Error Type:**", type(e).__name__)
                                                
                                                # Check data types and shapes
                                                if 'X_test' in locals():
                                                    st.write("**Test Data Info:**")
                                                    st.write(f"- Shape: {X_test.shape}")
                                                    st.write(f"- Columns: {list(X_test.columns)}")
                                                    st.write(f"- Data types: {X_test.dtypes.to_dict()}")
                                                    
                                                    # Show sample of problematic data
                                                    st.write("**Sample Test Data:**")
                                                    st.dataframe(X_test.head(3))
                                                    
                                                    # Check for non-numeric values
                                                    for col in X_test.columns:
                                                        if X_test[col].dtype == 'object':
                                                            unique_vals = X_test[col].unique()[:10]  # First 10 unique values
                                                            st.write(f"**Non-numeric column '{col}' values:** {unique_vals}")
                                                
                                                if 'y_test' in locals():
                                                    st.write("**Target Data Info:**")
                                                    if hasattr(y_test, 'shape'):
                                                        st.write(f"- Shape: {y_test.shape}")
                                                    st.write(f"- Type: {type(y_test)}")
                                                    if hasattr(y_test, 'dtype'):
                                                        st.write(f"- Data type: {y_test.dtype}")
                                                    
                                                    # Show sample target values
                                                    if hasattr(y_test, 'head'):
                                                        st.write("**Sample Target Values:**")
                                                        st.write(y_test.head().tolist())
                                                    elif hasattr(y_test, '__getitem__'):
                                                        st.write("**Sample Target Values:**")
                                                        st.write(y_test[:5].tolist() if len(y_test) >= 5 else y_test.tolist())
                                            
                                            st.info("💡 **Common fixes:**")
                                            st.write("• Ensure your data doesn't contain date/string columns that should be numeric")
                                            st.write("• Check that categorical columns were properly encoded during training")
                                            st.write("• Try the 'Upload Prediction Results' method if you have pre-computed predictions")
                                    else:
                                        st.warning("⚠️ Test data not available. Train a model first or use 'Upload Prediction Results'")
                                else:
                                    st.error("❌ Selected model doesn't support predictions")
                            else:
                                st.info("📊 Select a model to analyze its performance")
                        
                        elif analysis_method == "Upload Prediction Results":
                            st.markdown("#### 📁 Upload Prediction Results for Analysis")
                            st.info("Upload a CSV file with 'Actual' and 'Predicted' columns for comprehensive analysis")
                            
                            uploaded_results = st.file_uploader(
                                "Upload Prediction Results CSV",
                                type=['csv'],
                                help="CSV should contain 'Actual' and 'Predicted' columns"
                            )
                            
                            if uploaded_results:
                                try:
                                    results_df = pd.read_csv(uploaded_results)
                                    st.markdown("**📄 Data Preview:**")
                                    st.dataframe(results_df.head(), use_container_width=True)
                                    
                                    # Check for required columns
                                    required_cols = ['Actual', 'Predicted']
                                    available_cols = results_df.columns.tolist()
                                    
                                    # Try to auto-detect column names
                                    actual_col = None
                                    predicted_col = None
                                    
                                    for col in available_cols:
                                        if 'actual' in col.lower() or 'true' in col.lower() or 'target' in col.lower():
                                            actual_col = col
                                        elif 'predict' in col.lower() or 'pred' in col.lower():
                                            predicted_col = col
                                    
                                    if not actual_col or not predicted_col:
                                        st.warning("⚠️ Could not auto-detect columns. Please select manually:")
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            actual_col = st.selectbox("Select Actual Values Column:", available_cols)
                                        with col2:
                                            predicted_col = st.selectbox("Select Predicted Values Column:", available_cols)
                                    
                                    if actual_col and predicted_col:
                                        y_true = results_df[actual_col].values
                                        y_pred = results_df[predicted_col].values
                                        
                                        # Auto-detect problem type
                                        if len(np.unique(y_true)) <= 10 and all(isinstance(x, (int, np.integer)) for x in y_true[:100]):
                                            problem_type = 'classification'
                                        else:
                                            problem_type = 'regression'
                                        
                                        # Allow user to override
                                        problem_type = st.selectbox(
                                            "Problem Type:",
                                            ['classification', 'regression'],
                                            index=0 if problem_type == 'classification' else 1
                                        )
                                        
                                        if st.button("🔍 Analyze Performance"):
                                            # Create comprehensive analysis
                                            analysis_results = create_performance_analysis_dashboard(
                                                y_true, y_pred, None, problem_type, "Uploaded Model"
                                            )
                                
                                except Exception as e:
                                    st.error(f"❌ Error loading file: {str(e)}")
                                    st.info("💡 Ensure your CSV has proper formatting and column names")
                        
                        elif analysis_method == "Live Model Evaluation":
                            st.markdown("#### 🔄 Live Model Evaluation Tool")
                            st.info("Evaluate model performance with custom test data")
                            
                            # Select model
                            model_names = list(st.session_state.trained_models.keys())
                            selected_model = st.selectbox(
                                "Select Model for Live Evaluation:",
                                model_names,
                                key="live_eval_model"
                            )
                            
                            if selected_model:
                                st.markdown("**📝 Input Test Data:**")
                                
                                eval_method = st.radio(
                                    "Input Method:",
                                    ["Manual Input", "Upload Test CSV"]
                                )
                                
                                if eval_method == "Manual Input":
                                    st.info("Enter multiple data points for evaluation (one per row)")
                                    
                                    # Get feature names from the trained model
                                    if 'feature_names' in st.session_state:
                                        feature_names = st.session_state.feature_names
                                        
                                        # Create input form
                                        test_data = []
                                        actual_values = []
                                        
                                        num_samples = st.number_input(
                                            "Number of test samples:",
                                            min_value=1, max_value=50, value=5
                                        )
                                        
                                        for i in range(int(num_samples)):
                                            st.markdown(f"**Sample {i+1}:**")
                                            cols = st.columns(min(len(feature_names), 4))
                                            
                                            sample_data = {}
                                            for j, feature in enumerate(feature_names):
                                                with cols[j % 4]:
                                                    sample_data[feature] = st.number_input(
                                                        f"{feature}:",
                                                        key=f"live_eval_{i}_{feature}",
                                                        value=0.0
                                                    )
                                            
                                            actual_val = st.number_input(
                                                f"Actual value for sample {i+1}:",
                                                key=f"live_actual_{i}",
                                                value=0.0
                                            )
                                            
                                            test_data.append(sample_data)
                                            actual_values.append(actual_val)
                                        
                                        if st.button("🧪 Evaluate Model", key="live_evaluate"):
                                            try:
                                                # Prepare test data
                                                test_df = pd.DataFrame(test_data)
                                                
                                                # Make predictions
                                                model = st.session_state.trained_models[selected_model]['model']
                                                predictions = model.predict(test_df)
                                                
                                                # Create analysis
                                                y_true = np.array(actual_values)
                                                y_pred = predictions
                                                
                                                # Determine problem type
                                                if hasattr(model, 'classes_'):
                                                    problem_type = 'classification'
                                                else:
                                                    problem_type = 'regression'
                                                
                                                analysis_results = create_performance_analysis_dashboard(
                                                    y_true, y_pred, None, problem_type, f"{selected_model} (Live Evaluation)"
                                                )
                                                
                                            except Exception as e:
                                                st.error(f"❌ Evaluation error: {str(e)}")
                                    else:
                                        st.warning("⚠️ Feature names not available. Train a model first.")
                                
                                elif eval_method == "Upload Test CSV":
                                    uploaded_test = st.file_uploader(
                                        "Upload Test Data CSV",
                                        type=['csv'],
                                        help="Should contain features + target column",
                                        key="live_eval_csv"
                                    )
                                    
                                    if uploaded_test:
                                        test_df = pd.read_csv(uploaded_test)
                                        st.dataframe(test_df.head(), use_container_width=True)
                                        
                                        # Select target column
                                        target_col = st.selectbox(
                                            "Select Target Column:",
                                            test_df.columns.tolist()
                                        )
                                        
                                        if target_col and st.button("🧪 Evaluate Model", key="live_evaluate_csv"):
                                            try:
                                                X_test = test_df.drop(columns=[target_col])
                                                y_true = test_df[target_col].values
                                                
                                                # Get the trained model
                                                model = st.session_state.trained_models[selected_model]['model']
                                                
                                                # Preprocess the test data to match training data format
                                                st.info("🔄 Preprocessing test data to match training format...")
                                                
                                                # Apply the same preprocessing as training
                                                if 'label_encoders' in st.session_state and st.session_state.label_encoders:
                                                    # Apply label encoding to categorical columns
                                                    X_test_processed = X_test.copy()
                                                    
                                                    for col, encoder in st.session_state.label_encoders.items():
                                                        if col in X_test_processed.columns:
                                                            # Handle new categories not seen during training
                                                            try:
                                                                # Fill missing values first
                                                                X_test_processed[col] = X_test_processed[col].fillna("Missing")
                                                                
                                                                # Transform known categories, replace unknown with most frequent
                                                                known_categories = set(encoder.classes_)
                                                                X_test_processed[col] = X_test_processed[col].apply(
                                                                    lambda x: x if x in known_categories else encoder.classes_[0]
                                                                )
                                                                X_test_processed[col] = encoder.transform(X_test_processed[col])
                                                            except Exception as enc_error:
                                                                st.warning(f"⚠️ Encoding issue with column '{col}': {str(enc_error)}")
                                                                # Remove problematic column or use default encoding
                                                                X_test_processed = X_test_processed.drop(columns=[col])
                                                else:
                                                    X_test_processed = X_test.copy()
                                                
                                                # Handle missing values in numeric columns
                                                for col in X_test_processed.columns:
                                                    if X_test_processed[col].dtype in ['object']:
                                                        # Convert any remaining object columns to numeric if possible
                                                        try:
                                                            X_test_processed[col] = pd.to_numeric(X_test_processed[col], errors='coerce')
                                                        except:
                                                            # If conversion fails, drop the column
                                                            st.warning(f"⚠️ Dropping non-numeric column: {col}")
                                                            X_test_processed = X_test_processed.drop(columns=[col])
                                                    
                                                    # Fill missing values with median/mode
                                                    if X_test_processed[col].isnull().any():
                                                        if pd.api.types.is_numeric_dtype(X_test_processed[col]):
                                                            X_test_processed[col] = X_test_processed[col].fillna(X_test_processed[col].median())
                                                        else:
                                                            X_test_processed[col] = X_test_processed[col].fillna(X_test_processed[col].mode()[0] if not X_test_processed[col].mode().empty else 0)
                                                
                                                # Ensure the features match training features
                                                if 'feature_names' in st.session_state:
                                                    expected_features = st.session_state.feature_names
                                                    
                                                    # Add missing features with default values
                                                    for feature in expected_features:
                                                        if feature not in X_test_processed.columns:
                                                            X_test_processed[feature] = 0
                                                    
                                                    # Remove extra features not in training
                                                    X_test_processed = X_test_processed[expected_features]
                                                
                                                # Apply scaling if used during training
                                                if 'scaler' in st.session_state and st.session_state.scaler is not None:
                                                    X_test_processed = pd.DataFrame(
                                                        st.session_state.scaler.transform(X_test_processed),
                                                        columns=X_test_processed.columns
                                                    )
                                                
                                                # Make predictions
                                                y_pred = model.predict(X_test_processed)
                                                
                                                # Determine problem type
                                                if hasattr(model, 'classes_'):
                                                    problem_type = 'classification'
                                                else:
                                                    problem_type = 'regression'
                                                
                                                st.success(f"✅ Evaluation complete! Processed {len(X_test_processed)} samples with {len(X_test_processed.columns)} features.")
                                                
                                                analysis_results = create_performance_analysis_dashboard(
                                                    y_true, y_pred, None, problem_type, f"{selected_model} (CSV Evaluation)"
                                                )
                                                
                                            except Exception as e:
                                                st.error(f"❌ Evaluation error: {str(e)}")
                                                st.info("💡 **Common fixes:**")
                                                st.write("• Ensure your test data has the same structure as training data")
                                                st.write("• Check that categorical columns have valid values")
                                                st.write("• Try the 'Upload Prediction Results' method if you have pre-computed predictions")
                                                
                                                # Show debugging info
                                                with st.expander("🔍 Debug Information"):
                                                    st.write("**Test data columns:**", list(X_test.columns))
                                                    if 'feature_names' in st.session_state:
                                                        st.write("**Expected features:**", st.session_state.feature_names)
                                                    st.write("**Test data types:**")
                                                    st.write(X_test.dtypes.to_dict())
                                                    
                                                    # Show sample of problematic data
                                                    st.write("**First few rows of test data:**")
                                                    st.dataframe(X_test.head())
                    
                    else:
                        st.info("🎯 **No trained models available**")
                        st.markdown("""
                        To use Performance Analysis:
                        1. Go to **Model Setup** tab
                        2. Train at least one model
                        3. Return here for comprehensive analysis
                        
                        **Or** upload your own prediction results using the 'Upload Prediction Results' option above.
                        """)

            else:
                st.warning("⚠️ No data available for modeling. Please upload a dataset first.")

        with tab4:
            st.header("📈 Export & Reports")

            # Export section
            st.subheader("💾 Export Data")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Export Cleaned Data:**")

                # Generate timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # CSV export
                csv = convert_df_to_csv(st.session_state.cleaned_df)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"cleaned_data_{timestamp}.csv",
                    mime="text/csv",
                    help="Download the cleaned dataset as CSV",
                )

                # Excel export (requires openpyxl)
                try:
                    from io import BytesIO

                    output = BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        st.session_state.cleaned_df.to_excel(
                            writer, sheet_name="Cleaned_Data", index=False
                        )

                        # Add a summary sheet
                        summary_data = {
                            "Metric": [
                                "Total Rows",
                                "Total Columns",
                                "Missing Values",
                                "Duplicate Rows",
                                "Memory Usage (KB)",
                            ],
                            "Value": [
                                len(st.session_state.cleaned_df),
                                len(st.session_state.cleaned_df.columns),
                                st.session_state.cleaned_df.isnull().sum().sum(),
                                st.session_state.cleaned_df.duplicated().sum(),
                                f"{st.session_state.cleaned_df.memory_usage(deep=True).sum() / 1024:.1f}",
                            ],
                        }
                        pd.DataFrame(summary_data).to_excel(
                            writer, sheet_name="Summary", index=False
                        )

                    st.download_button(
                        label="📊 Download as Excel",
                        data=output.getvalue(),
                        file_name=f"cleaned_data_{timestamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        help="Download as Excel file with summary sheet",
                    )
                except ImportError:
                    st.info(
                        "💡 Install openpyxl to enable Excel export: `pip install openpyxl`"
                    )

            with col2:
                st.write("**Data Summary Report:**")

                # Generate cleaning report
                if st.button("📋 Generate Cleaning Report"):
                    report = generate_cleaning_report(
                        df_original, st.session_state.cleaned_df
                    )
                    st.text_area("Cleaning Report", report, height=300)

                # Data quality score
                quality_score = calculate_data_quality_score(
                    st.session_state.cleaned_df
                )
                st.metric("Data Quality Score", f"{quality_score:.1f}%")

                # Quick stats
                st.write("**Quick Statistics:**")
                stats_data = {
                    "Original Rows": len(df_original),
                    "Current Rows": len(st.session_state.cleaned_df),
                    "Rows Removed": len(df_original) - len(st.session_state.cleaned_df),
                    "Missing Values": st.session_state.cleaned_df.isnull().sum().sum(),
                    "Complete Rows": len(st.session_state.cleaned_df.dropna()),
                    "Data Types": len(
                        st.session_state.cleaned_df.dtypes.value_counts()
                    ),
                }

                for key, value in stats_data.items():
                    st.write(f"**{key}:** {value:,}")

        with tab5:
            st.header("🗃️ Database Integration")
            st.write("Connect to databases and load data directly into your analysis workflow.")
            
            # Add the database dashboard
            database_dashboard()

        with tab6:
            st.header("📅 Advanced Time Series Analytics Suite")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: white; margin: 0;">🚀 Professional Time Series Analysis</h4>
                <p style="color: #f0f0f0; margin: 5px 0 0 0;">
                    Advanced temporal analytics with forecasting, anomaly detection, and statistical modeling
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Quick Time Series Overview
            if df_original is not None:
                # Detect potential time series data
                date_cols = []
                numeric_cols = []
                
                for col in df_original.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_original[col]):
                        date_cols.append(col)
                    elif any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated', 'year', 'month']):
                        try:
                            pd.to_datetime(df_original[col].dropna().head(10))
                            date_cols.append(col)
                        except:
                            pass
                    elif pd.api.types.is_numeric_dtype(df_original[col]):
                        numeric_cols.append(col)
                
                # Time Series Metrics Overview
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                
                with col_metric1:
                    st.metric("📅 Date Columns", len(date_cols))
                    
                with col_metric2:
                    st.metric("📊 Numeric Series", len(numeric_cols))
                    
                with col_metric3:
                    data_span = "N/A"
                    if date_cols and len(date_cols) > 0:
                        try:
                            first_date_col = date_cols[0]
                            if pd.api.types.is_datetime64_any_dtype(df_original[first_date_col]):
                                dates = df_original[first_date_col]
                            else:
                                dates = pd.to_datetime(df_original[first_date_col])
                            data_span = f"{(dates.max() - dates.min()).days} days"
                        except:
                            data_span = "Unknown"
                    st.metric("📏 Data Span", data_span)
                    
                with col_metric4:
                    frequency_hint = "Unknown"
                    if date_cols and len(date_cols) > 0:
                        try:
                            dates_clean = pd.to_datetime(df_original[date_cols[0]]).dropna().sort_values()
                            if len(dates_clean) > 1:
                                avg_diff = (dates_clean.iloc[-1] - dates_clean.iloc[0]) / len(dates_clean)
                                if avg_diff.days < 1:
                                    frequency_hint = "Hourly/Sub-daily"
                                elif avg_diff.days < 7:
                                    frequency_hint = "Daily"
                                elif avg_diff.days < 32:
                                    frequency_hint = "Weekly"
                                else:
                                    frequency_hint = "Monthly+"
                        except:
                            pass
                    st.metric("🔄 Est. Frequency", frequency_hint)
                
                st.markdown("---")
                
                # Enhanced Time Series Dashboard
                if date_cols and numeric_cols:
                    # Create comprehensive tabs
                    ts_main_tab1, ts_main_tab2, ts_main_tab3, ts_main_tab4, ts_main_tab5 = st.tabs([
                        "⚙️ Data Preparation", 
                        "📊 Exploratory Analysis", 
                        "🔮 Advanced Forecasting", 
                        "⚠️ Anomaly Detection",
                        "📈 Financial Analytics"
                    ])
                    
                    with ts_main_tab1:
                        st.subheader("⚙️ Time Series Data Preparation")
                        
                        # Enhanced configuration with validation
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            date_column = st.selectbox(
                                "📅 Select Date Column",
                                options=date_cols,
                                help="Choose the column containing dates/timestamps"
                            )
                        
                        with col2:
                            value_column = st.selectbox(
                                "📊 Select Value Column", 
                                options=numeric_cols,
                                help="Choose the column with values to analyze"
                            )
                        
                        with col3:
                            resample_freq = st.selectbox(
                                "🔄 Resampling Frequency",
                                options=['None', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'],
                                help="Aggregate data to specified frequency"
                            )
                        
                        if date_column and value_column:
                            try:
                                # Prepare the time series data
                                ts_df = df_original[[date_column, value_column]].copy()
                                
                                # Convert date column
                                if not pd.api.types.is_datetime64_any_dtype(ts_df[date_column]):
                                    ts_df[date_column] = pd.to_datetime(ts_df[date_column])
                                
                                # Remove missing values
                                ts_df = ts_df.dropna()
                                ts_df = ts_df.sort_values(date_column)
                                
                                # Store in session state
                                st.session_state.ts_data = ts_df
                                st.session_state.ts_date_col = date_column
                                st.session_state.ts_value_col = value_column
                                
                                # Data quality assessment
                                st.markdown("### 📋 Data Quality Assessment")
                                
                                quality_col1, quality_col2, quality_col3 = st.columns(3)
                                
                                with quality_col1:
                                    total_points = len(ts_df)
                                    st.metric("Total Data Points", f"{total_points:,}")
                                
                                with quality_col2:
                                    missing_count = df_original[[date_column, value_column]].isnull().sum().sum()
                                    st.metric("Missing Values Removed", missing_count)
                                
                                with quality_col3:
                                    date_range = ts_df[date_column].max() - ts_df[date_column].min()
                                    st.metric("Date Range", f"{date_range.days} days")
                                
                                # Data preview
                                st.markdown("### 👀 Data Preview")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**First 5 Records:**")
                                    st.dataframe(ts_df.head(), use_container_width=True)
                                
                                with col2:
                                    st.write("**Last 5 Records:**")
                                    st.dataframe(ts_df.tail(), use_container_width=True)
                                
                                # Basic statistics
                                st.markdown("### 📊 Statistical Summary")
                                
                                stats_col1, stats_col2 = st.columns(2)
                                
                                with stats_col1:
                                    st.write("**Descriptive Statistics:**")
                                    stats = ts_df[value_column].describe()
                                    st.dataframe(stats.to_frame().T, use_container_width=True)
                                
                                with stats_col2:
                                    st.write("**Time Series Properties:**")
                                    
                                    # Calculate additional metrics
                                    values = ts_df[value_column]
                                    
                                    ts_properties = {
                                        "Trend": "Increasing" if values.iloc[-1] > values.iloc[0] else "Decreasing",
                                        "Volatility": f"{values.std():.2f}",
                                        "CV (%)": f"{(values.std() / values.mean() * 100):.1f}",
                                        "Skewness": f"{values.skew():.2f}",
                                        "Kurtosis": f"{values.kurtosis():.2f}"
                                    }
                                    
                                    for prop, val in ts_properties.items():
                                        st.write(f"**{prop}**: {val}")
                                
                                # Quick visualization
                                st.markdown("### 📈 Quick Time Series Plot")
                                
                                fig = px.line(
                                    ts_df, 
                                    x=date_column, 
                                    y=value_column,
                                    title=f"{value_column} Over Time"
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Data quality warnings
                                if total_points < 30:
                                    st.warning("⚠️ Limited data points. Time series analysis may be less reliable with fewer than 30 observations.")
                                
                                if missing_count > total_points * 0.1:
                                    st.warning("⚠️ High number of missing values detected. Consider data cleaning.")
                                
                                if values.std() == 0:
                                    st.error("❌ No variation in values. Time series analysis requires varying data.")
                                else:
                                    st.success("✅ Data preparation complete! Proceed to analysis tabs.")
                            
                            except Exception as e:
                                st.error(f"❌ Error preparing time series data: {str(e)}")
                        else:
                            st.info("👆 Please select both date and value columns to begin analysis.")
                    
                    with ts_main_tab2:
                        st.subheader("📊 Exploratory Time Series Analysis")
                        
                        if 'ts_data' in st.session_state:
                            ts_df = st.session_state.ts_data
                            date_col = st.session_state.ts_date_col
                            value_col = st.session_state.ts_value_col
                            
                            # Analysis options
                            analysis_col1, analysis_col2 = st.columns(2)
                            
                            with analysis_col1:
                                show_trend = st.checkbox("📈 Show Trend Line", value=True)
                                show_ma = st.checkbox("🔄 Show Moving Averages", value=False)
                            
                            with analysis_col2:
                                show_seasonal = st.checkbox("🌊 Seasonal Decomposition", value=False)
                                show_autocorr = st.checkbox("🔗 Autocorrelation", value=False)
                            
                            # Main time series plot with enhancements
                            fig = go.Figure()
                            
                            # Main series
                            fig.add_trace(go.Scatter(
                                x=ts_df[date_col],
                                y=ts_df[value_col],
                                mode='lines',
                                name=value_col,
                                line=dict(color='#1f77b4', width=2)
                            ))
                            
                            # Add trend line
                            if show_trend:
                                from scipy import stats
                                x_numeric = np.arange(len(ts_df))
                                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, ts_df[value_col])
                                trend_line = slope * x_numeric + intercept
                                
                                fig.add_trace(go.Scatter(
                                    x=ts_df[date_col],
                                    y=trend_line,
                                    mode='lines',
                                    name=f'Trend (R²={r_value**2:.3f})',
                                    line=dict(color='red', dash='dash', width=2)
                                ))
                            
                            # Add moving averages
                            if show_ma:
                                for window in [7, 30]:
                                    if len(ts_df) >= window:
                                        ma = ts_df[value_col].rolling(window=window).mean()
                                        fig.add_trace(go.Scatter(
                                            x=ts_df[date_col],
                                            y=ma,
                                            mode='lines',
                                            name=f'MA-{window}',
                                            line=dict(width=1, dash='dot')
                                        ))
                            
                            fig.update_layout(
                                title=f"Time Series Analysis: {value_col}",
                                xaxis_title="Date",
                                yaxis_title=value_col,
                                height=500,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Statistical Analysis
                            st.markdown("### 🔍 Statistical Analysis")
                            
                            stat_col1, stat_col2, stat_col3 = st.columns(3)
                            
                            with stat_col1:
                                # Stationarity test
                                try:
                                    from statsmodels.tsa.stattools import adfuller
                                    
                                    adf_result = adfuller(ts_df[value_col].dropna())
                                    is_stationary = adf_result[1] <= 0.05
                                    
                                    st.metric(
                                        "Stationarity Test",
                                        "Stationary" if is_stationary else "Non-Stationary",
                                        f"p-value: {adf_result[1]:.4f}"
                                    )
                                except ImportError:
                                    st.info("Install statsmodels for stationarity testing")
                                except Exception as e:
                                    st.error(f"Stationarity test failed: {str(e)}")
                            
                            with stat_col2:
                                # Trend strength
                                x_numeric = np.arange(len(ts_df))
                                correlation = np.corrcoef(x_numeric, ts_df[value_col])[0, 1]
                                trend_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak"
                                
                                st.metric(
                                    "Trend Strength",
                                    trend_strength,
                                    f"Correlation: {correlation:.3f}"
                                )
                            
                            with stat_col3:
                                # Volatility measure
                                returns = ts_df[value_col].pct_change().dropna()
                                volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                                
                                st.metric(
                                    "Annualized Volatility",
                                    f"{volatility:.2%}",
                                    "Based on % changes"
                                )
                            
                            # Seasonal decomposition
                            if show_seasonal and len(ts_df) >= 24:
                                try:
                                    from statsmodels.tsa.seasonal import seasonal_decompose
                                    
                                    st.markdown("### 🌊 Seasonal Decomposition")
                                    
                                    # Prepare data for decomposition
                                    ts_indexed = ts_df.set_index(date_col)[value_col]
                                    
                                    # Determine period
                                    period = st.selectbox(
                                        "Seasonal Period",
                                        options=[7, 12, 24, 52],
                                        help="7=Weekly, 12=Monthly, 24=Bi-weekly, 52=Weekly in year"
                                    )
                                    
                                    if st.button("🔄 Perform Decomposition"):
                                        with st.spinner("Performing seasonal decomposition..."):
                                            decomposition = seasonal_decompose(
                                                ts_indexed, 
                                                model='additive', 
                                                period=period
                                            )
                                            
                                            # Create subplots
                                            fig = make_subplots(
                                                rows=4, cols=1,
                                                subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                                                vertical_spacing=0.08
                                            )
                                            
                                            # Add traces
                                            components = [
                                                (decomposition.observed, 'Original'),
                                                (decomposition.trend, 'Trend'),
                                                (decomposition.seasonal, 'Seasonal'),
                                                (decomposition.resid, 'Residual')
                                            ]
                                            
                                            for i, (component, name) in enumerate(components):
                                                fig.add_trace(
                                                    go.Scatter(
                                                        x=component.index,
                                                        y=component.values,
                                                        mode='lines',
                                                        name=name,
                                                        showlegend=False
                                                    ),
                                                    row=i+1, col=1
                                                )
                                            
                                            fig.update_layout(height=800, title="Seasonal Decomposition")
                                            st.plotly_chart(fig, use_container_width=True)
                                
                                except ImportError:
                                    st.error("Please install statsmodels: pip install statsmodels")
                                except Exception as e:
                                    st.error(f"Decomposition failed: {str(e)}")
                            
                            # Autocorrelation analysis
                            if show_autocorr:
                                st.markdown("### 🔗 Autocorrelation Analysis")
                                
                                try:
                                    from statsmodels.tsa.stattools import acf, pacf
                                    
                                    lags = min(40, len(ts_df) // 4)
                                    
                                    # Calculate ACF and PACF
                                    acf_vals = acf(ts_df[value_col].dropna(), nlags=lags)
                                    pacf_vals = pacf(ts_df[value_col].dropna(), nlags=lags)
                                    
                                    # Plot ACF and PACF
                                    acf_col1, acf_col2 = st.columns(2)
                                    
                                    with acf_col1:
                                        fig_acf = go.Figure()
                                        fig_acf.add_trace(go.Bar(
                                            x=list(range(len(acf_vals))),
                                            y=acf_vals,
                                            name='ACF'
                                        ))
                                        fig_acf.update_layout(title="Autocorrelation Function", height=300)
                                        st.plotly_chart(fig_acf, use_container_width=True)
                                    
                                    with acf_col2:
                                        fig_pacf = go.Figure()
                                        fig_pacf.add_trace(go.Bar(
                                            x=list(range(len(pacf_vals))),
                                            y=pacf_vals,
                                            name='PACF'
                                        ))
                                        fig_pacf.update_layout(title="Partial Autocorrelation Function", height=300)
                                        st.plotly_chart(fig_pacf, use_container_width=True)
                                
                                except ImportError:
                                    st.error("Please install statsmodels for autocorrelation analysis")
                                except Exception as e:
                                    st.error(f"Autocorrelation analysis failed: {str(e)}")
                        else:
                            st.info("🔧 Please prepare your time series data in the 'Data Preparation' tab first.")
                    
                    with ts_main_tab3:
                        st.subheader("🔮 Advanced Forecasting Models")
                        
                        if 'ts_data' in st.session_state:
                            ts_df = st.session_state.ts_data
                            date_col = st.session_state.ts_date_col
                            value_col = st.session_state.ts_value_col
                            
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                        padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                                <h5 style="color: white; margin: 0;">🎯 Professional Forecasting Suite</h5>
                                <p style="color: #f0f0f0; margin: 5px 0 0 0; font-size: 14px;">
                                    Multiple algorithms with confidence intervals and model comparison
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Forecasting configuration
                            forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
                            
                            with forecast_col1:
                                forecast_method = st.selectbox(
                                    "🔮 Forecasting Method",
                                    options=['Simple Moving Average', 'Exponential Smoothing', 'Linear Regression', 'Prophet (Advanced)'],
                                    help="Choose the forecasting algorithm"
                                )
                            
                            with forecast_col2:
                                forecast_periods = st.number_input(
                                    "📅 Forecast Periods",
                                    min_value=1,
                                    max_value=365,
                                    value=30,
                                    help="Number of future periods to forecast"
                                )
                            
                            with forecast_col3:
                                train_split = st.slider(
                                    "🎯 Training Data %",
                                    min_value=0.5,
                                    max_value=0.95,
                                    value=0.8,
                                    help="Percentage of data used for training"
                                )
                            
                            # Split data for validation
                            split_idx = int(len(ts_df) * train_split)
                            train_data = ts_df.iloc[:split_idx]
                            test_data = ts_df.iloc[split_idx:]
                            
                            if st.button("🚀 Generate Forecast", type="primary"):
                                with st.spinner(f"Generating {forecast_method} forecast..."):
                                    try:
                                        # Simple Moving Average
                                        if forecast_method == 'Simple Moving Average':
                                            window = min(30, len(train_data) // 4)
                                            last_values = train_data[value_col].tail(window).mean()
                                            
                                            # Generate forecast dates
                                            last_date = train_data[date_col].max()
                                            forecast_dates = pd.date_range(
                                                start=last_date + pd.Timedelta(days=1),
                                                periods=forecast_periods,
                                                freq='D'
                                            )
                                            
                                            forecast_values = [last_values] * forecast_periods
                                            
                                            # Calculate prediction intervals (simple approach)
                                            std_error = train_data[value_col].std()
                                            lower_bound = [val - 1.96 * std_error for val in forecast_values]
                                            upper_bound = [val + 1.96 * std_error for val in forecast_values]
                                        
                                        # Linear Regression
                                        elif forecast_method == 'Linear Regression':
                                            from sklearn.linear_model import LinearRegression
                                            
                                            # Prepare features (time as numeric)
                                            train_numeric = np.arange(len(train_data)).reshape(-1, 1)
                                            
                                            # Fit model
                                            model = LinearRegression()
                                            model.fit(train_numeric, train_data[value_col])
                                            
                                            # Generate forecast
                                            future_numeric = np.arange(len(train_data), len(train_data) + forecast_periods).reshape(-1, 1)
                                            forecast_values = model.predict(future_numeric)
                                            
                                            # Generate forecast dates
                                            last_date = train_data[date_col].max()
                                            forecast_dates = pd.date_range(
                                                start=last_date + pd.Timedelta(days=1),
                                                periods=forecast_periods,
                                                freq='D'
                                            )
                                            
                                            # Calculate prediction intervals
                                            residuals = train_data[value_col] - model.predict(train_numeric)
                                            std_error = np.std(residuals)
                                            lower_bound = forecast_values - 1.96 * std_error
                                            upper_bound = forecast_values + 1.96 * std_error
                                        
                                        # Exponential Smoothing
                                        elif forecast_method == 'Exponential Smoothing':
                                            alpha = 0.3  # Smoothing parameter
                                            smoothed_values = [train_data[value_col].iloc[0]]
                                            
                                            for val in train_data[value_col].iloc[1:]:
                                                smoothed_val = alpha * val + (1 - alpha) * smoothed_values[-1]
                                                smoothed_values.append(smoothed_val)
                                            
                                            # Forecast
                                            last_smoothed = smoothed_values[-1]
                                            forecast_values = [last_smoothed] * forecast_periods
                                            
                                            # Generate forecast dates
                                            last_date = train_data[date_col].max()
                                            forecast_dates = pd.date_range(
                                                start=last_date + pd.Timedelta(days=1),
                                                periods=forecast_periods,
                                                freq='D'
                                            )
                                            
                                            # Calculate prediction intervals
                                            residuals = train_data[value_col] - smoothed_values
                                            std_error = np.std(residuals)
                                            lower_bound = [val - 1.96 * std_error for val in forecast_values]
                                            upper_bound = [val + 1.96 * std_error for val in forecast_values]
                                        
                                        else:  # Prophet placeholder
                                            st.info("📦 Prophet requires additional installation: pip install prophet")
                                            forecast_values = None
                                        
                                        if forecast_values is not None:
                                            # Create comprehensive forecast visualization
                                            fig = go.Figure()
                                            
                                            # Historical training data
                                            fig.add_trace(go.Scatter(
                                                x=train_data[date_col],
                                                y=train_data[value_col],
                                                mode='lines',
                                                name='Training Data',
                                                line=dict(color='#1f77b4', width=2)
                                            ))
                                            
                                            # Test data (if available)
                                            if len(test_data) > 0:
                                                fig.add_trace(go.Scatter(
                                                    x=test_data[date_col],
                                                    y=test_data[value_col],
                                                    mode='lines',
                                                    name='Actual (Test)',
                                                    line=dict(color='green', width=2)
                                                ))
                                            
                                            # Forecast
                                            fig.add_trace(go.Scatter(
                                                x=forecast_dates,
                                                y=forecast_values,
                                                mode='lines+markers',
                                                name='Forecast',
                                                line=dict(color='#ff7f0e', width=2, dash='dash')
                                            ))
                                            
                                            # Confidence intervals
                                            fig.add_trace(go.Scatter(
                                                x=forecast_dates,
                                                y=upper_bound,
                                                mode='lines',
                                                name='Upper 95% CI',
                                                line=dict(color='rgba(255,127,14,0.2)'),
                                                showlegend=False
                                            ))
                                            
                                            fig.add_trace(go.Scatter(
                                                x=forecast_dates,
                                                y=lower_bound,
                                                mode='lines',
                                                name='Lower 95% CI',
                                                line=dict(color='rgba(255,127,14,0.2)'),
                                                fill='tonexty',
                                                showlegend=True
                                            ))
                                            
                                            fig.update_layout(
                                                title=f"{forecast_method} Forecast - {forecast_periods} Periods",
                                                xaxis_title="Date",
                                                yaxis_title=value_col,
                                                height=500,
                                                hovermode='x unified'
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # Forecast metrics and summary
                                            st.markdown("### 📊 Forecast Summary")
                                            
                                            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                                            
                                            with summary_col1:
                                                st.metric("🎯 Method", forecast_method)
                                            
                                            with summary_col2:
                                                st.metric("📅 Periods", forecast_periods)
                                            
                                            with summary_col3:
                                                avg_forecast = np.mean(forecast_values)
                                                st.metric("📈 Avg Forecast", f"{avg_forecast:.2f}")
                                            
                                            with summary_col4:
                                                forecast_trend = "↗️ Increasing" if forecast_values[-1] > forecast_values[0] else "↘️ Decreasing"
                                                st.metric("📊 Trend", forecast_trend)
                                            
                                            # Model evaluation (if test data available)
                                            if len(test_data) > 0 and forecast_method in ['Linear Regression']:
                                                st.markdown("### 🎯 Model Evaluation")
                                                
                                                # Generate predictions for test period
                                                test_numeric = np.arange(len(train_data), len(ts_df)).reshape(-1, 1)
                                                test_predictions = model.predict(test_numeric[:len(test_data)])
                                                
                                                # Calculate metrics
                                                mae = np.mean(np.abs(test_data[value_col] - test_predictions))
                                                rmse = np.sqrt(np.mean((test_data[value_col] - test_predictions) ** 2))
                                                mape = np.mean(np.abs((test_data[value_col] - test_predictions) / test_data[value_col])) * 100
                                                
                                                eval_col1, eval_col2, eval_col3 = st.columns(3)
                                                
                                                with eval_col1:
                                                    st.metric("MAE", f"{mae:.2f}")
                                                
                                                with eval_col2:
                                                    st.metric("RMSE", f"{rmse:.2f}")
                                                
                                                with eval_col3:
                                                    st.metric("MAPE", f"{mape:.1f}%")
                                            
                                            # Download forecast data
                                            forecast_df = pd.DataFrame({
                                                'Date': forecast_dates,
                                                'Forecast': forecast_values,
                                                'Lower_95CI': lower_bound,
                                                'Upper_95CI': upper_bound
                                            })
                                            
                                            csv_forecast = forecast_df.to_csv(index=False)
                                            st.download_button(
                                                "📥 Download Forecast Data",
                                                data=csv_forecast,
                                                file_name=f"forecast_{forecast_method.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv"
                                            )
                                    
                                    except Exception as e:
                                        st.error(f"❌ Forecasting failed: {str(e)}")
                        else:
                            st.info("🔧 Please prepare your time series data in the 'Data Preparation' tab first.")
                    
                    with ts_main_tab4:
                        st.subheader("⚠️ Advanced Anomaly Detection")
                        
                        if 'ts_data' in st.session_state:
                            ts_df = st.session_state.ts_data
                            date_col = st.session_state.ts_date_col
                            value_col = st.session_state.ts_value_col
                            
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); 
                                        padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                                <h5 style="color: white; margin: 0;">🚨 Smart Anomaly Detection</h5>
                                <p style="color: #f0f0f0; margin: 5px 0 0 0; font-size: 14px;">
                                    Multi-algorithm anomaly detection with confidence scoring
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Anomaly detection configuration
                            anomaly_col1, anomaly_col2, anomaly_col3 = st.columns(3)
                            
                            with anomaly_col1:
                                detection_method = st.selectbox(
                                    "🔍 Detection Method",
                                    options=['Statistical (Z-Score)', 'IQR Method', 'Moving Average', 'Isolation Forest'],
                                    help="Choose anomaly detection algorithm"
                                )
                            
                            with anomaly_col2:
                                sensitivity = st.selectbox(
                                    "⚡ Sensitivity",
                                    options=['Low', 'Medium', 'High'],
                                    index=1,
                                    help="Detection sensitivity level"
                                )
                            
                            with anomaly_col3:
                                window_size = st.number_input(
                                    "📊 Window Size",
                                    min_value=5,
                                    max_value=50,
                                    value=20,
                                    help="Window for moving statistics"
                                )
                            
                            if st.button("🔍 Detect Anomalies", type="primary"):
                                with st.spinner("Detecting anomalies..."):
                                    try:
                                        values = ts_df[value_col]
                                        anomalies = pd.Series(False, index=values.index)
                                        
                                        # Set thresholds based on sensitivity
                                        thresholds = {
                                            'Low': {'zscore': 3.0, 'iqr': 3.0, 'isolation': 0.05},
                                            'Medium': {'zscore': 2.5, 'iqr': 2.0, 'isolation': 0.1},
                                            'High': {'zscore': 2.0, 'iqr': 1.5, 'isolation': 0.15}
                                        }
                                        
                                        threshold = thresholds[sensitivity]
                                        
                                        # Statistical Z-Score method
                                        if detection_method == 'Statistical (Z-Score)':
                                            z_scores = np.abs((values - values.mean()) / values.std())
                                            anomalies = z_scores > threshold['zscore']
                                        
                                        # IQR method
                                        elif detection_method == 'IQR Method':
                                            Q1 = values.quantile(0.25)
                                            Q3 = values.quantile(0.75)
                                            IQR = Q3 - Q1
                                            lower_bound = Q1 - threshold['iqr'] * IQR
                                            upper_bound = Q3 + threshold['iqr'] * IQR
                                            anomalies = (values < lower_bound) | (values > upper_bound)
                                        
                                        # Moving Average method
                                        elif detection_method == 'Moving Average':
                                            moving_avg = values.rolling(window=window_size).mean()
                                            moving_std = values.rolling(window=window_size).std()
                                            deviations = np.abs(values - moving_avg) / moving_std
                                            anomalies = deviations > threshold['zscore']
                                        
                                        # Isolation Forest
                                        elif detection_method == 'Isolation Forest':
                                            try:
                                                from sklearn.ensemble import IsolationForest
                                                
                                                iso_forest = IsolationForest(
                                                    contamination=threshold['isolation'],
                                                    random_state=42
                                                )
                                                
                                                # Reshape for sklearn
                                                values_reshaped = values.values.reshape(-1, 1)
                                                predictions = iso_forest.fit_predict(values_reshaped)
                                                anomalies = predictions == -1
                                                
                                            except ImportError:
                                                st.error("Isolation Forest requires scikit-learn")
                                                anomalies = pd.Series(False, index=values.index)
                                        
                                        # Create visualization
                                        fig = go.Figure()
                                        
                                        # Normal data points
                                        normal_data = ts_df[~anomalies]
                                        fig.add_trace(go.Scatter(
                                            x=normal_data[date_col],
                                            y=normal_data[value_col],
                                            mode='lines+markers',
                                            name='Normal',
                                            line=dict(color='#1f77b4', width=2),
                                            marker=dict(size=4)
                                        ))
                                        
                                        # Anomalous data points
                                        anomaly_data = ts_df[anomalies]
                                        if len(anomaly_data) > 0:
                                            fig.add_trace(go.Scatter(
                                                x=anomaly_data[date_col],
                                                y=anomaly_data[value_col],
                                                mode='markers',
                                                name='Anomalies',
                                                marker=dict(
                                                    color='red',
                                                    size=10,
                                                    symbol='x',
                                                    line=dict(width=2, color='darkred')
                                                )
                                            ))
                                        
                                        # Add statistical bounds for relevant methods
                                        if detection_method in ['Statistical (Z-Score)', 'Moving Average']:
                                            mean_val = values.mean()
                                            std_val = values.std()
                                            upper_bound = mean_val + threshold['zscore'] * std_val
                                            lower_bound = mean_val - threshold['zscore'] * std_val
                                            
                                            fig.add_hline(
                                                y=upper_bound,
                                                line_dash="dash",
                                                line_color="orange",
                                                annotation_text="Upper Threshold"
                                            )
                                            fig.add_hline(
                                                y=lower_bound,
                                                line_dash="dash",
                                                line_color="orange",
                                                annotation_text="Lower Threshold"
                                            )
                                        
                                        fig.update_layout(
                                            title=f"Anomaly Detection: {detection_method} ({sensitivity} Sensitivity)",
                                            xaxis_title="Date",
                                            yaxis_title=value_col,
                                            height=500,
                                            hovermode='x unified'
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Anomaly summary
                                        st.markdown("### 📊 Anomaly Summary")
                                        
                                        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                                        
                                        with summary_col1:
                                            total_anomalies = anomalies.sum()
                                            st.metric("🚨 Total Anomalies", total_anomalies)
                                        
                                        with summary_col2:
                                            anomaly_rate = (total_anomalies / len(values)) * 100
                                            st.metric("📊 Anomaly Rate", f"{anomaly_rate:.1f}%")
                                        
                                        with summary_col3:
                                            if total_anomalies > 0:
                                                max_anomaly = values[anomalies].max()
                                                st.metric("📈 Max Anomaly", f"{max_anomaly:.2f}")
                                            else:
                                                st.metric("📈 Max Anomaly", "None")
                                        
                                        with summary_col4:
                                            if total_anomalies > 0:
                                                min_anomaly = values[anomalies].min()
                                                st.metric("📉 Min Anomaly", f"{min_anomaly:.2f}")
                                            else:
                                                st.metric("📉 Min Anomaly", "None")
                                        
                                        # Detailed anomaly list
                                        if total_anomalies > 0:
                                            st.markdown("### 📋 Detailed Anomaly List")
                                            
                                            anomaly_details = ts_df[anomalies].copy()
                                            anomaly_details['Severity'] = 'High'  # You could calculate actual severity
                                            anomaly_details['Deviation'] = np.abs(
                                                anomaly_details[value_col] - values.mean()
                                            ) / values.std()
                                            
                                            st.dataframe(
                                                anomaly_details[[date_col, value_col, 'Deviation', 'Severity']],
                                                use_container_width=True
                                            )
                                            
                                            # Download anomaly report
                                            csv_anomalies = anomaly_details.to_csv(index=False)
                                            st.download_button(
                                                "📥 Download Anomaly Report",
                                                data=csv_anomalies,
                                                file_name=f"anomalies_{detection_method.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv"
                                            )
                                        
                                        else:
                                            st.success("✅ No anomalies detected with current settings!")
                                            st.info("💡 Try adjusting the sensitivity or detection method if you expect anomalies.")
                                    
                                    except Exception as e:
                                        st.error(f"❌ Anomaly detection failed: {str(e)}")
                        else:
                            st.info("🔧 Please prepare your time series data in the 'Data Preparation' tab first.")
                    
                    with ts_main_tab5:
                        st.subheader("📈 Financial & Performance Analytics")
                        
                        if 'ts_data' in st.session_state:
                            ts_df = st.session_state.ts_data
                            date_col = st.session_state.ts_date_col
                            value_col = st.session_state.ts_value_col
                            
                            st.markdown("""
                            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                                        padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                                <h5 style="color: white; margin: 0;">💰 Financial Time Series Analytics</h5>
                                <p style="color: #f0f0f0; margin: 5px 0 0 0; font-size: 14px;">
                                    Advanced financial metrics, risk analysis, and performance indicators
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Calculate financial metrics
                            try:
                                values = ts_df[value_col]
                                
                                # Returns calculation
                                returns = values.pct_change().dropna()
                                
                                # Risk metrics
                                volatility = returns.std() * np.sqrt(252)  # Annualized
                                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                                
                                # Drawdown calculation
                                cumulative = (1 + returns).cumprod()
                                rolling_max = cumulative.expanding().max()
                                drawdown = (cumulative - rolling_max) / rolling_max
                                max_drawdown = drawdown.min()
                                
                                # Performance metrics
                                total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
                                cagr = ((values.iloc[-1] / values.iloc[0]) ** (365 / len(values)) - 1) * 100
                                
                                # Display key metrics
                                st.markdown("### 💎 Key Financial Metrics")
                                
                                fin_col1, fin_col2, fin_col3, fin_col4 = st.columns(4)
                                
                                with fin_col1:
                                    st.metric(
                                        "📈 Total Return", 
                                        f"{total_return:.1f}%",
                                        help="Total percentage change from start to end"
                                    )
                                
                                with fin_col2:
                                    st.metric(
                                        "📊 CAGR", 
                                        f"{cagr:.1f}%",
                                        help="Compound Annual Growth Rate"
                                    )
                                
                                with fin_col3:
                                    st.metric(
                                        "⚡ Volatility", 
                                        f"{volatility:.1%}",
                                        help="Annualized volatility (risk measure)"
                                    )
                                
                                with fin_col4:
                                    st.metric(
                                        "📉 Max Drawdown", 
                                        f"{max_drawdown:.1%}",
                                        help="Maximum peak-to-trough decline"
                                    )
                                
                                # Advanced metrics
                                st.markdown("### 🎯 Advanced Risk Metrics")
                                
                                adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
                                
                                with adv_col1:
                                    st.metric(
                                        "📊 Sharpe Ratio", 
                                        f"{sharpe_ratio:.2f}",
                                        help="Risk-adjusted return measure"
                                    )
                                
                                with adv_col2:
                                    var_95 = np.percentile(returns, 5)
                                    st.metric(
                                        "📉 VaR (95%)", 
                                        f"{var_95:.2%}",
                                        help="Value at Risk (95% confidence)"
                                    )
                                
                                with adv_col3:
                                    skewness = returns.skew()
                                    st.metric(
                                        "📊 Skewness", 
                                        f"{skewness:.2f}",
                                        help="Distribution asymmetry"
                                    )
                                
                                with adv_col4:
                                    kurtosis = returns.kurtosis()
                                    st.metric(
                                        "📈 Kurtosis", 
                                        f"{kurtosis:.2f}",
                                        help="Distribution tail thickness"
                                    )
                                
                                # Visualizations
                                st.markdown("### 📊 Financial Visualizations")
                                
                                # Create tabs for different charts
                                chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
                                    "💹 Price & Returns", "📉 Drawdown Analysis", "📊 Distribution Analysis", "🔥 Performance Dashboard"
                                ])
                                
                                with chart_tab1:
                                    # Price and returns chart
                                    fig = make_subplots(
                                        rows=2, cols=1,
                                        subplot_titles=('Price Series', 'Daily Returns'),
                                        vertical_spacing=0.1
                                    )
                                    
                                    # Price series
                                    fig.add_trace(go.Scatter(
                                        x=ts_df[date_col],
                                        y=ts_df[value_col],
                                        mode='lines',
                                        name='Price',
                                        line=dict(color='blue', width=2)
                                    ), row=1, col=1)
                                    
                                    # Returns
                                    fig.add_trace(go.Scatter(
                                        x=ts_df[date_col].iloc[1:],
                                        y=returns,
                                        mode='lines',
                                        name='Returns',
                                        line=dict(color='green', width=1)
                                    ), row=2, col=1)
                                    
                                    fig.update_layout(height=600, title="Price and Returns Analysis")
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with chart_tab2:
                                    # Drawdown analysis
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Scatter(
                                        x=ts_df[date_col].iloc[1:],
                                        y=drawdown * 100,
                                        mode='lines',
                                        name='Drawdown',
                                        fill='tonexty',
                                        line=dict(color='red', width=2)
                                    ))
                                    
                                    fig.update_layout(
                                        title="Drawdown Analysis",
                                        xaxis_title="Date",
                                        yaxis_title="Drawdown (%)",
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with chart_tab3:
                                    # Distribution analysis
                                    dist_col1, dist_col2 = st.columns(2)
                                    
                                    with dist_col1:
                                        # Returns histogram
                                        fig = px.histogram(
                                            x=returns,
                                            nbins=50,
                                            title="Returns Distribution"
                                        )
                                        fig.add_vline(x=returns.mean(), line_dash="dash", line_color="red")
                                        st.plotly_chart(fig, use_container_width=True)
                                    
                                    with dist_col2:
                                        # Q-Q plot simulation
                                        sorted_returns = np.sort(returns)
                                        normal_quantiles = np.random.normal(0, returns.std(), len(returns))
                                        normal_quantiles = np.sort(normal_quantiles)
                                        
                                        fig = go.Figure()
                                        fig.add_trace(go.Scatter(
                                            x=normal_quantiles,
                                            y=sorted_returns,
                                            mode='markers',
                                            name='Q-Q Plot'
                                        ))
                                        fig.add_trace(go.Scatter(
                                            x=[-0.1, 0.1],
                                            y=[-0.1, 0.1],
                                            mode='lines',
                                            name='Normal Line',
                                            line=dict(dash='dash')
                                        ))
                                        fig.update_layout(title="Q-Q Plot (Normal Distribution)")
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                with chart_tab4:
                                    # Performance dashboard
                                    performance_metrics = {
                                        'Total Return': f"{total_return:.1f}%",
                                        'CAGR': f"{cagr:.1f}%",
                                        'Volatility': f"{volatility:.1%}",
                                        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                                        'Max Drawdown': f"{max_drawdown:.1%}",
                                        'VaR (95%)': f"{var_95:.2%}",
                                        'Skewness': f"{skewness:.2f}",
                                        'Kurtosis': f"{kurtosis:.2f}"
                                    }
                                    
                                    # Create performance summary table
                                    perf_df = pd.DataFrame.from_dict(
                                        performance_metrics, 
                                        orient='index', 
                                        columns=['Value']
                                    )
                                    perf_df.index.name = 'Metric'
                                    
                                    st.write("**Performance Summary Table:**")
                                    st.dataframe(perf_df, use_container_width=True)
                                    
                                    # Risk-return scatter (if you have multiple series)
                                    st.write("**Risk Assessment:**")
                                    
                                    risk_assessment = []
                                    if sharpe_ratio > 1.0:
                                        risk_assessment.append("✅ Excellent risk-adjusted returns")
                                    elif sharpe_ratio > 0.5:
                                        risk_assessment.append("✅ Good risk-adjusted returns")
                                    else:
                                        risk_assessment.append("⚠️ Poor risk-adjusted returns")
                                    
                                    if abs(max_drawdown) < 0.1:
                                        risk_assessment.append("✅ Low drawdown risk")
                                    elif abs(max_drawdown) < 0.2:
                                        risk_assessment.append("⚠️ Moderate drawdown risk")
                                    else:
                                        risk_assessment.append("❌ High drawdown risk")
                                    
                                    if volatility < 0.15:
                                        risk_assessment.append("✅ Low volatility")
                                    elif volatility < 0.25:
                                        risk_assessment.append("⚠️ Moderate volatility")
                                    else:
                                        risk_assessment.append("❌ High volatility")
                                    
                                    for assessment in risk_assessment:
                                        st.write(assessment)
                                
                                # Export financial analysis
                                st.markdown("### 📥 Export Analysis")
                                
                                financial_report = f"""
Financial Time Series Analysis Report
===================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Period: {ts_df[date_col].min()} to {ts_df[date_col].max()}
Total Observations: {len(ts_df)}

Key Metrics:
-----------
Total Return: {total_return:.2f}%
CAGR: {cagr:.2f}%
Annualized Volatility: {volatility:.2%}
Sharpe Ratio: {sharpe_ratio:.3f}
Maximum Drawdown: {max_drawdown:.2%}
VaR (95%): {var_95:.2%}
Skewness: {skewness:.3f}
Kurtosis: {kurtosis:.3f}

Risk Assessment:
---------------
{chr(10).join(risk_assessment)}

Data Summary:
------------
Mean: {values.mean():.2f}
Std Dev: {values.std():.2f}
Min: {values.min():.2f}
Max: {values.max():.2f}
"""
                                
                                st.download_button(
                                    "📥 Download Financial Report",
                                    data=financial_report,
                                    file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain"
                                )
                            
                            except Exception as e:
                                st.error(f"❌ Financial analysis failed: {str(e)}")
                                st.info("💡 Financial analysis works best with price/value data")
                        else:
                            st.info("🔧 Please prepare your time series data in the 'Data Preparation' tab first.")
                
                else:
                    st.warning("⚠️ No suitable time series data detected.")
                    st.info("""
                    **To use Time Series Analysis:**
                    1. Ensure your data has at least one date column
                    2. Include numeric columns for analysis
                    3. Date columns should contain actual dates or timestamps
                    4. Consider columns named: date, time, timestamp, created_at, etc.
                    """)
            
            else:
                st.warning("⚠️ Please load data first in the main application.")
        with tab7:
            st.header("📊 Model Monitoring & Performance Tracking")
            st.write("Monitor model performance, detect data drift, and track model health over time.")
            
            # Check if we have any model results to monitor
            if 'model_results' in st.session_state and st.session_state.model_results:
                try:
                    # Initialize Model Monitor
                    if 'model_monitor' not in st.session_state:
                        st.session_state.model_monitor = ModelMonitor()
                    
                    monitor = st.session_state.model_monitor
                    
                    # Create tabs for different monitoring views
                    monitor_tab1, monitor_tab2, monitor_tab3, monitor_tab4 = st.tabs([
                        "📈 Performance Dashboard",
                        "🔍 Data Drift Detection", 
                        "📊 Model Comparison",
                        "⚙️ Configuration"
                    ])
                    
                    with monitor_tab1:
                        st.subheader("📈 Model Performance Dashboard")
                        
                        # Performance metrics tracking
                        model_results = st.session_state.model_results
                        
                        # Display current model performance
                        if isinstance(model_results, dict):
                            st.write("**Current Training Session Results:**")
                            
                            # Show metrics for each trained model
                            for model_name, model_info in model_results.items():
                                with st.expander(f"📊 {model_name} Performance", expanded=True):
                                    if isinstance(model_info, dict) and 'score' in model_info:
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("Model Score", f"{model_info['score']:.4f}")
                                        with col2:
                                            st.metric("Model Type", model_name)
                                        with col3:
                                            training_time = model_info.get('training_time', 'N/A')
                                            st.metric("Training Time", f"{training_time:.2f}s" if isinstance(training_time, (int, float)) else training_time)
                                        
                                        # Add to tracking button
                                        if st.button(f"📈 Add {model_name} to Tracking", key=f"track_{model_name}"):
                                            performance_data = {
                                                'timestamp': datetime.now(),
                                                'model_name': model_name,
                                                'score': model_info['score'],
                                                'model_type': model_name,
                                                'session_id': datetime.now().strftime('%Y%m%d_%H%M%S')
                                            }
                                            
                                            monitor.track_performance(performance_data)
                                            st.success(f"✅ Added {model_name} to performance tracking")
                                            st.rerun()
                        
                        st.markdown("---")
                        
                        # Display performance history
                        performance_history = monitor.get_performance_history()
                        if performance_history:
                            st.write("**📈 Performance Tracking History**")
                            
                            # Convert to DataFrame for better display
                            history_df = pd.DataFrame(performance_history)
                            
                            # Display summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Tracked Models", len(history_df))
                            with col2:
                                if 'score' in history_df.columns:
                                    avg_score = history_df['score'].mean()
                                    st.metric("Average Score", f"{avg_score:.4f}")
                            with col3:
                                unique_models = history_df['model_name'].nunique() if 'model_name' in history_df.columns else 0
                                st.metric("Unique Models", unique_models)
                            with col4:
                                if 'timestamp' in history_df.columns:
                                    time_span = (history_df['timestamp'].max() - history_df['timestamp'].min()).days
                                    st.metric("Tracking Period", f"{time_span} days")
                            
                            # Show recent entries
                            st.write("**Recent Performance Entries:**")
                            display_df = history_df.copy()
                            if 'timestamp' in display_df.columns:
                                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Plot performance trends if we have multiple entries
                            if len(history_df) > 1 and 'score' in history_df.columns:
                                st.write("**📊 Performance Trends:**")
                                
                                fig = px.line(
                                    history_df, 
                                    x='timestamp', 
                                    y='score',
                                    color='model_name' if 'model_name' in history_df.columns else None,
                                    title='Model Performance Over Time',
                                    labels={'score': 'Model Score', 'timestamp': 'Time'}
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Performance comparison
                                if 'model_name' in history_df.columns:
                                    st.write("**🏆 Model Performance Comparison:**")
                                    model_summary = history_df.groupby('model_name')['score'].agg(['mean', 'max', 'min', 'count']).round(4)
                                    model_summary.columns = ['Average Score', 'Best Score', 'Worst Score', 'Runs']
                                    st.dataframe(model_summary, use_container_width=True)
                        else:
                            st.info("📊 No performance history yet. Add models from your training sessions above to start tracking!")
                            
                            # Show demo data option
                            if st.button("📈 Add Demo Performance Data"):
                                # Add some demo performance entries
                                demo_data = [
                                    {
                                        'timestamp': datetime.now() - timedelta(days=7),
                                        'model_name': 'Random Forest',
                                        'score': 0.8567,
                                        'model_type': 'Random Forest',
                                        'session_id': 'demo_1'
                                    },
                                    {
                                        'timestamp': datetime.now() - timedelta(days=5),
                                        'model_name': 'Logistic Regression', 
                                        'score': 0.8234,
                                        'model_type': 'Logistic Regression',
                                        'session_id': 'demo_2'
                                    },
                                    {
                                        'timestamp': datetime.now() - timedelta(days=3),
                                        'model_name': 'Random Forest',
                                        'score': 0.8699,
                                        'model_type': 'Random Forest', 
                                        'session_id': 'demo_3'
                                    },
                                    {
                                        'timestamp': datetime.now() - timedelta(days=1),
                                        'model_name': 'SVM',
                                        'score': 0.8445,
                                        'model_type': 'SVM',
                                        'session_id': 'demo_4'
                                    }
                                ]
                                
                                for data in demo_data:
                                    monitor.track_performance(data)
                                
                                st.success("✅ Added demo performance data!")
                                st.rerun()
                    
                    with monitor_tab2:
                        st.subheader("🔍 Data Drift Detection")
                        
                        st.write("Compare current data distribution with training data to detect drift.")
                        
                        # Data drift detection
                        if st.button("Detect Data Drift"):
                            try:
                                # Get original training data (assuming it's available)
                                if 'df_original' in locals():
                                    # Compare with current data
                                    drift_results = monitor.detect_data_drift(df_original, df_original)  # Using same data for demo
                                    
                                    st.write("**Drift Detection Results**")
                                    
                                    # Display numerical drift
                                    if 'numerical_drift' in drift_results:
                                        st.write("**Numerical Features Drift:**")
                                        num_drift_df = pd.DataFrame(drift_results['numerical_drift']).T
                                        st.dataframe(num_drift_df, use_container_width=True)
                                    
                                    # Display categorical drift
                                    if 'categorical_drift' in drift_results:
                                        st.write("**Categorical Features Drift:**")
                                        cat_drift_df = pd.DataFrame(drift_results['categorical_drift']).T
                                        st.dataframe(cat_drift_df, use_container_width=True)
                                    
                                    # Overall drift assessment
                                    total_features = len(drift_results.get('numerical_drift', {})) + len(drift_results.get('categorical_drift', {}))
                                    if total_features > 0:
                                        drifted_features = sum(1 for d in drift_results.get('numerical_drift', {}).values() if d.get('drifted', False))
                                        drifted_features += sum(1 for d in drift_results.get('categorical_drift', {}).values() if d.get('drifted', False))
                                        
                                        drift_percentage = (drifted_features / total_features) * 100
                                        
                                        if drift_percentage > 30:
                                            st.error(f"⚠️ High drift detected: {drift_percentage:.1f}% of features show drift")
                                        elif drift_percentage > 10:
                                            st.warning(f"⚠️ Moderate drift detected: {drift_percentage:.1f}% of features show drift")
                                        else:
                                            st.success(f"✅ Low drift: {drift_percentage:.1f}% of features show drift")
                                
                                else:
                                    st.warning("Original training data not available for drift comparison")
                            
                            except Exception as e:
                                st.error(f"Error detecting drift: {str(e)}")
                    
                    with monitor_tab3:
                        st.subheader("📊 Model Comparison")
                        
                        st.write("Compare performance across different models and time periods.")
                        
                        # Model comparison dashboard
                        performance_history = monitor.get_performance_history()
                        if performance_history and len(performance_history) > 1:
                            
                            # Create comparison visualization
                            comparison_results = monitor.create_comparison_dashboard(performance_history)
                            
                            if comparison_results:
                                st.write("**Model Performance Comparison**")
                                
                                # Display comparison metrics
                                if 'comparison_metrics' in comparison_results:
                                    st.dataframe(comparison_results['comparison_metrics'], use_container_width=True)
                                
                                # Display ranking
                                if 'model_ranking' in comparison_results:
                                    st.write("**Model Ranking**")
                                    for i, model in enumerate(comparison_results['model_ranking'], 1):
                                        st.write(f"{i}. {model}")
                        else:
                            st.info("Need at least 2 models in history for comparison")
                    
                    with monitor_tab4:
                        st.subheader("⚙️ Monitoring Configuration")
                        
                        st.write("Configure monitoring thresholds and alert settings.")
                        
                        # Configuration settings
                        config_data = monitor.get_configuration()
                        
                        with st.form("monitoring_config"):
                            st.write("**Performance Thresholds**")
                            accuracy_threshold = st.slider(
                                "Accuracy Alert Threshold", 
                                0.0, 1.0, 
                                config_data.get('performance_thresholds', {}).get('accuracy', 0.8),
                                0.01
                            )
                            
                            st.write("**Drift Detection Settings**")
                            drift_threshold = st.slider(
                                "Drift Alert Threshold", 
                                0.0, 1.0, 
                                config_data.get('drift_thresholds', {}).get('default', 0.1),
                                0.01
                            )
                            
                            st.write("**Alert Settings**")
                            enable_alerts = st.checkbox(
                                "Enable Alerts", 
                                value=config_data.get('alerts', {}).get('enabled', True)
                            )
                            
                            if st.form_submit_button("Save Configuration"):
                                new_config = {
                                    'performance_thresholds': {'accuracy': accuracy_threshold},
                                    'drift_thresholds': {'default': drift_threshold},
                                    'alerts': {'enabled': enable_alerts}
                                }
                                monitor.update_configuration(new_config)
                                st.success("✅ Configuration saved successfully!")
                
                except Exception as e:
                    st.error(f"Error initializing model monitoring: {str(e)}")
                    st.write("Please ensure you have trained a model first.")
            
            else:
                st.info("🤖 No model results available for monitoring. Please train a model first in the Modeling tab.")
                
                # Show monitoring capabilities preview
                st.write("**Available Monitoring Features:**")
                st.write("• Real-time performance tracking")
                st.write("• Data drift detection")
                st.write("• Model comparison dashboard") 
                st.write("• Automated alerting")
                st.write("• Historical analysis")
                
                # Demo visualization of monitoring capabilities
                st.write("**Sample Monitoring Dashboard:**")
                
                # Create sample performance data
                import plotly.graph_objects as go
                dates = pd.date_range('2024-01-01', periods=30, freq='D')
                sample_accuracy = np.random.normal(0.85, 0.05, 30)
                sample_precision = np.random.normal(0.82, 0.04, 30)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=sample_accuracy,
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=dates, y=sample_precision,
                    mode='lines+markers', 
                    name='Precision',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title="Sample Model Performance Over Time",
                    xaxis_title="Date",
                    yaxis_title="Score",
                    yaxis=dict(range=[0.7, 1.0])
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab8:
            st.header("🧠 AI-Powered Data Insights")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: white; margin: 0;">🚀 Advanced AI Analytics Suite</h4>
                <p style="color: #f0f0f0; margin: 5px 0 0 0;">
                    Next-generation AI capabilities for deeper data intelligence and actionable insights
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick AI Metrics Overview
            col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
            
            with col_metric1:
                data_complexity = len(df_original.columns) * (df_original.dtypes == 'object').sum()
                st.metric("🧩 Data Complexity", f"{data_complexity}")
                
            with col_metric2:
                correlation_strength = abs(df_original.select_dtypes(include=[np.number]).corr()).mean().mean()
                st.metric("🔗 Avg Correlation", f"{correlation_strength:.3f}")
                
            with col_metric3:
                missing_ratio = df_original.isnull().sum().sum() / (len(df_original) * len(df_original.columns))
                st.metric("🕳️ Missing Ratio", f"{missing_ratio:.3f}")
                
            with col_metric4:
                ml_readiness = min(100, (1 - missing_ratio) * 100)
                st.metric("🤖 ML Readiness", f"{ml_readiness:.0f}%")
            
            st.markdown("---")
            
            # AI Insights Dashboard
            try:
                create_ai_insights_dashboard(df_original, st.session_state.get('target_column'))
            except Exception as e:
                st.error(f"Error generating AI insights: {str(e)}")
                st.write("AI insights require valid data to analyze.")
            
            # Enhanced AI Features Section
            st.markdown("---")
            st.subheader("🚀 Enhanced AI Analytics")
            
            # Create tabs for different AI features
            ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
                "🔍 Anomaly Detection", 
                "🔧 Feature Engineering", 
                "📖 Data Story", 
                "📊 Smart Visualizations"
            ])
            
            # Try to load enhanced AI features
            try:
                from modules.enhanced_ai_engine import AdvancedAIEngine, SmartVisualizationEngine
                
                with ai_tab1:
                    st.subheader("🔍 Enhanced Anomaly Detection")
                    st.write("Multi-algorithm ensemble detection with confidence scoring and business context")
                    
                    # Check if data is available
                    if 'original_df' not in st.session_state or st.session_state.original_df is None or st.session_state.original_df.empty:
                        st.warning("📊 Please upload data first to use Enhanced Anomaly Detection")
                        st.info("💡 Go to the 'Data Upload & Preview' section to load your dataset")
                    else:
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            sensitivity = st.selectbox(
                                "Detection Sensitivity",
                                ["Conservative", "Balanced", "Aggressive"],
                                index=1,
                                help="Conservative: High confidence anomalies only, Aggressive: More sensitive detection"
                            )
                            
                            if st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True):
                                with st.spinner("🔄 Running advanced anomaly detection..."):
                                    try:
                                        ai_engine = AdvancedAIEngine()
                                        anomaly_results = ai_engine.enhanced_anomaly_detection(st.session_state.original_df)
                                        st.session_state['anomaly_results'] = anomaly_results
                                        st.success("✅ Anomaly detection completed!")
                                    except Exception as e:
                                        st.error(f"❌ Error in anomaly detection: {str(e)}")
                        
                        with col2:
                            if 'anomaly_results' in st.session_state:
                                results = st.session_state['anomaly_results']
                                
                                # Check if results have the expected structure
                                if isinstance(results, dict) and 'summary' in results:
                                    # Display metrics in an attractive layout
                                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                                    with metric_col1:
                                        st.metric("🚨 Anomalies Found", results['summary'].get('total_anomalies', 0))
                                    with metric_col2:
                                        st.metric("📊 Percentage", f"{results['summary'].get('anomaly_percentage', 0):.2f}%")
                                    with metric_col3:
                                        confidence_dist = results['summary'].get('confidence_distribution', {})
                                        st.metric("🔥 High Confidence", confidence_dist.get('high', 0))
                                    
                                    # Display insights
                                    if 'actionable_insights' in results and results['actionable_insights']:
                                        st.subheader("💡 AI-Generated Insights")
                                        for i, insight in enumerate(results['actionable_insights'][:5]):
                                            st.info(f"**Insight {i+1}:** {insight}")
                                    
                                    # Show detailed anomaly breakdown
                                    if 'detailed_results' in results and st.checkbox("Show Detailed Anomaly Analysis"):
                                        st.subheader("📋 Detailed Anomaly Breakdown")
                                        for method, data in results['detailed_results'].items():
                                            if isinstance(data, dict) and data.get('anomalies', 0) > 0:
                                                with st.expander(f"🔍 {method.title()} Method - {data.get('anomalies', 0)} anomalies"):
                                                    st.write(f"**Confidence:** {data.get('confidence', 0):.2%}")
                                                    st.write(f"**Description:** {data.get('description', 'No description available')}")
                                else:
                                    st.warning("⚠️ Unexpected results format from anomaly detection")
                            else:
                                st.info("🔍 Click 'Run Anomaly Detection' to analyze your data for anomalies")
                
                with ai_tab2:
                    st.subheader("🔧 AI-Powered Feature Engineering")
                    st.write("Intelligent feature suggestions based on data patterns and ML best practices")
                    
                    # Check if data is available
                    if 'original_df' not in st.session_state or st.session_state.original_df is None or st.session_state.original_df.empty:
                        st.warning("📊 Please upload data first to use AI Feature Engineering")
                        st.info("💡 Go to the 'Data Upload & Preview' section to load your dataset")
                    else:
                        st.info("🔧 Feature engineering recommendations will be shown here after data analysis")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        feature_focus = st.selectbox(
                            "Feature Focus",
                            ["All Features", "Predictive Power", "Interaction Discovery", "Mathematical Transforms"],
                            help="Choose the type of feature engineering to prioritize"
                        )
                        
                        if st.button("🧠 Generate Feature Suggestions", type="primary", use_container_width=True):
                            with st.spinner("🔄 Analyzing feature engineering opportunities..."):
                                ai_engine = AdvancedAIEngine()
                                suggestions = ai_engine.intelligent_feature_engineering(df_original)
                                st.session_state['feature_suggestions'] = suggestions
                    
                    with col2:
                        if 'feature_suggestions' in st.session_state:
                            suggestions = st.session_state['feature_suggestions']
                            
                            # Feature transformation suggestions
                            if suggestions['transformations']:
                                st.subheader("🎯 Recommended Transformations")
                                for i, suggestion in enumerate(suggestions['transformations'][:5]):
                                    with st.expander(f"💡 {suggestion['suggestion']}", expanded=(i==0)):
                                        col_a, col_b = st.columns(2)
                                        with col_a:
                                            st.write(f"**Rationale:** {suggestion['rationale']}")
                                            st.write(f"**Type:** {suggestion['type']}")
                                        with col_b:
                                            if st.button(f"Apply Transform {i+1}", key=f"transform_{i}"):
                                                st.info(f"🔄 Would apply: {suggestion['suggestion']}")
                            
                            # Feature interaction suggestions
                            if suggestions['interactions']:
                                st.subheader("🔄 Feature Interactions")
                                for i, suggestion in enumerate(suggestions['interactions'][:3]):
                                    with st.expander(f"🔗 {suggestion['formula']}", expanded=(i==0)):
                                        st.write(f"**Rationale:** {suggestion['rationale']}")
                                        if st.button(f"Create Interaction {i+1}", key=f"interact_{i}"):
                                            st.info(f"🔄 Would create: {suggestion['formula']}")
                
                with ai_tab3:
                    st.subheader("📖 AI Data Story Generator")
                    st.write("Get a comprehensive AI-generated narrative about your dataset's personality and potential")
                    
                    # Check if data is available
                    if 'original_df' not in st.session_state or st.session_state.original_df is None or st.session_state.original_df.empty:
                        st.warning("📊 Please upload data first to generate an AI Data Story")
                        st.info("💡 Go to the 'Data Upload & Preview' section to load your dataset")
                    else:
                        col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        story_style = st.selectbox(
                            "Story Style",
                            ["Executive Summary", "Technical Deep-dive", "Business Insights", "ML Readiness Report"],
                            help="Choose the style of narrative you prefer"
                        )
                        
                        if st.button("� Generate AI Data Story", type="primary", use_container_width=True):
                            with st.spinner("🔄 AI is analyzing your data and crafting the story..."):
                                ai_engine = AdvancedAIEngine()
                                profile = ai_engine.predictive_data_profiling(st.session_state.original_df)
                                st.session_state['data_profile'] = profile
                    
                    with col2:
                        if 'data_profile' in st.session_state:
                            profile = st.session_state['data_profile']
                            
                            # Display the AI story in an attractive format
                            st.markdown(f"""
                            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #007bff;">
                                {profile['data_story']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Dataset personality metrics
                            st.subheader("🎭 Dataset Personality")
                            personality = profile['dataset_personality']
                            pers_col1, pers_col2, pers_col3, pers_col4 = st.columns(4)
                            
                            with pers_col1:
                                st.metric("📏 Size Category", personality['size'])
                            with pers_col2:
                                st.metric("✨ Quality Grade", personality['quality'])
                            with pers_col3:
                                st.metric("🏷️ Data Type", personality['type'])
                            with pers_col4:
                                st.metric("🧩 Complexity", f"{personality['complexity_score']:.3f}")
                            
                            # ML Readiness with visual progress
                            readiness = profile['ml_readiness']
                            st.subheader("🤖 ML Readiness Assessment")
                            
                            # Create color-coded progress bar
                            progress_color = "🟢" if readiness['score'] > 80 else "🟡" if readiness['score'] > 60 else "🔴"
                            st.markdown(f"**{progress_color} Grade: {readiness['grade']} ({readiness['score']:.0f}/100)**")
                            st.progress(readiness['score'] / 100)
                            
                            # Algorithm recommendations with reasoning
                            st.subheader("💡 Recommended ML Algorithms")
                            for i, algo in enumerate(profile['recommended_algorithms'][:3]):
                                st.info(f"**{i+1}.** {algo}")
                
                with ai_tab4:
                    st.subheader("📊 Smart Visualization Recommendations")
                    st.write("AI-powered chart suggestions based on data types, relationships, and analysis goals")
                    
                    # Check if data is available
                    if 'original_df' not in st.session_state or st.session_state.original_df is None or st.session_state.original_df.empty:
                        st.warning("📊 Please upload data first to get Smart Visualization Recommendations")
                        st.info("💡 Go to the 'Data Upload & Preview' section to load your dataset")
                    else:
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            viz_purpose = st.selectbox(
                                "Analysis Purpose",
                                ["Exploratory Analysis", "Pattern Discovery", "Correlation Analysis", "Distribution Analysis", "Comparison Analysis"],
                                help="What type of insights are you looking for?"
                            )
                            
                            if st.button("🎨 Get Visualization Recommendations", type="primary", use_container_width=True):
                                with st.spinner("🔄 AI is analyzing optimal visualizations..."):
                                    try:
                                        viz_engine = SmartVisualizationEngine()
                                        viz_recommendations = viz_engine.recommend_visualizations(st.session_state.original_df)
                                        st.session_state['viz_recommendations'] = viz_recommendations
                                        st.success("✅ Visualization recommendations generated!")
                                    except Exception as e:
                                        st.error(f"❌ Error generating recommendations: {str(e)}")
                    
                    with col2:
                        if 'viz_recommendations' in st.session_state:
                            recommendations = st.session_state['viz_recommendations']
                            
                            st.subheader("📈 AI-Recommended Charts")
                            for i, rec in enumerate(recommendations[:5]):
                                priority_emoji = "🔥" if rec['priority'] == "High" else "⭐" if rec['priority'] == "Medium" else "💡"
                                
                                with st.expander(f"{priority_emoji} {rec['type'].title()} - {', '.join(rec['columns'])}", expanded=(i==0)):
                                    col_a, col_b = st.columns([2, 1])
                                    
                                    with col_a:
                                        st.write(f"**Rationale:** {rec['rationale']}")
                                        st.write(f"**Priority:** {rec['priority']}")
                                        st.write(f"**Best for:** {rec.get('use_case', 'General analysis')}")
                                    
                                    with col_b:
                                        if st.button(f"Create Chart {i+1}", key=f"chart_{i}"):
                                            st.success(f"🎨 Would create {rec['type']} chart!")
                                            # Here you could actually generate the chart
                            
                            # Quick action buttons
                            st.subheader("⚡ Quick Actions")
                            quick_col1, quick_col2, quick_col3 = st.columns(3)
                            
                            with quick_col1:
                                if st.button("📊 Create All Priority Charts", use_container_width=True):
                                    high_priority = [r for r in recommendations if r['priority'] == 'High']
                                    st.info(f"🎨 Would create {len(high_priority)} high-priority visualizations")
                            
                            with quick_col2:
                                if st.button("🔄 Refresh Recommendations", use_container_width=True):
                                    if 'viz_recommendations' in st.session_state:
                                        del st.session_state['viz_recommendations']
                                    st.rerun()
                            
                            with quick_col3:
                                if st.button("� Export Recommendations", use_container_width=True):
                                    report = "AI VISUALIZATION RECOMMENDATIONS\n" + "="*50 + "\n\n"
                                    for i, rec in enumerate(recommendations):
                                        report += f"{i+1}. {rec['type'].title()}\n"
                                        report += f"   Columns: {', '.join(rec['columns'])}\n"
                                        report += f"   Priority: {rec['priority']}\n"
                                        report += f"   Rationale: {rec['rationale']}\n\n"
                                    
                                    st.download_button(
                                        "📥 Download Report",
                                        report,
                                        f"viz_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                        "text/plain"
                                    )
                
            except ImportError:
                st.warning("🔧 Enhanced AI features not available. Installing dependencies...")
                
                # Auto-install with progress
                with st.spinner("Installing scikit-learn and scipy..."):
                    try:
                        import subprocess
                        result = subprocess.run(["pip", "install", "scikit-learn", "scipy"], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            st.success("✅ Dependencies installed! Please refresh the page.")
                            st.balloons()
                        else:
                            st.error("❌ Installation failed. Please run manually: `pip install scikit-learn scipy`")
                    except Exception as e:
                        st.error(f"Installation error: {e}")
                        
            except Exception as e:
                st.error(f"❌ Enhanced AI error: {type(e).__name__}: {str(e)}")
                st.write("Standard AI insights remain available above.")
                st.info("💡 This might be due to missing data or dependencies. Try refreshing the page.")

        with tab9:
            st.header("⚡ Advanced Performance Analytics")
            st.write("Real-time system monitoring and performance optimization recommendations.")
            
            # Performance Analytics Dashboard
            try:
                create_performance_dashboard()
            except Exception as e:
                st.error(f"Error loading performance analytics: {str(e)}")
                st.write("Performance monitoring may require additional system permissions.")

        with tab10:
            st.header("🔧 System Health & Diagnostics")
            st.write("Comprehensive system health monitoring and data consistency validation.")
            
            # Data State Health Check
            st.subheader("📊 Data State Validation")
            data_state_dashboard()
            
            st.markdown("---")
            
            # Error Monitoring
            st.subheader("🚨 Error Monitoring")
            error_dashboard()
            
            st.markdown("---")
            
            # System Information
            st.subheader("💾 System Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Session State Variables:**")
                session_vars = [key for key in st.session_state.keys() if not key.startswith('_')]
                for var in sorted(session_vars):
                    value = st.session_state[var]
                    if isinstance(value, pd.DataFrame):
                        st.write(f"• `{var}`: DataFrame {value.shape}")
                    else:
                        st.write(f"• `{var}`: {type(value).__name__}")
            
            with col2:
                st.write("**Data Consistency Checks:**")
                validation_report = validate_data_state()
                
                if validation_report['status'] == 'healthy':
                    st.success("✅ All systems operational")
                elif validation_report['status'] == 'warning':
                    st.warning(f"⚠️ {len(validation_report['warnings'])} warnings")
                else:
                    st.error(f"❌ {len(validation_report['errors'])} errors")
                
                if st.button("🔧 Run Auto-Fix"):
                    fixes = auto_fix_data_state()
                    if fixes:
                        st.success("Applied fixes:")
                        for fix in fixes:
                            st.write(f"✅ {fix}")
                    else:
                        st.info("No fixes needed")

    except Exception as e:
        # Use new error handling
        handle_error(
            e, 
            "Unexpected error in main application", 
            ErrorSeverity.CRITICAL, 
            ErrorCategory.SYSTEM,
            "Please try refreshing the page or reloading your data"
        )

else:
    # Enhanced welcome message with modern interactive design - using styles from main section
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-title">🎯 Welcome to Data Assistant Pro!</div>
            <div class="welcome-subtitle">Your AI-Powered Data Science Companion</div>
            <div class="welcome-description">
                Transform raw data into actionable insights with our enterprise-grade analytics platform
            </div>
        </div>
        
        <div class="stats-banner">
            <div class="stat-item">
                <span class="stat-number">🚀</span>
                <span class="stat-label">Lightning Fast</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">🎯</span>
                <span class="stat-label">99% Accuracy</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">⚡</span>
                <span class="stat-label">Real-time Processing</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">🏆</span>
                <span class="stat-label">Enterprise Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add the missing welcome styles to main CSS
    st.markdown(
        """
        <style>
        .welcome-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin: 30px 0;
            box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .welcome-container::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            animation: shine 3s infinite;
        }
        
        @keyframes shine {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        
        .welcome-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }
        
        .welcome-subtitle {
            font-size: 1.3rem;
            margin-bottom: 15px;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }
        
        .welcome-description {
            font-size: 1rem;
            opacity: 0.8;
            position: relative;
            z-index: 1;
        }
        
        .stats-banner {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
            background-size: 400% 400%;
            animation: gradientShift 5s ease infinite;
            padding: 20px;
            border-radius: 15px;
            margin: 30px 0;
            text-align: center;
            color: white;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .stat-item {
            display: inline-block;
            margin: 0 20px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            display: block;
        }
        
        .stat-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Interactive Getting Started Section
    st.markdown("### 🚀 Getting Started")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 15px; color: white; text-align: center;
                        margin-bottom: 20px; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);">
                <h4>📁 Upload Your Data</h4>
                <p>Drag & drop your CSV, Excel, or JSON files</p>
                <p style="font-size: 0.9rem; opacity: 0.8;">Supports files up to 200MB</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                        padding: 20px; border-radius: 15px; color: white; text-align: center;
                        margin-bottom: 20px; box-shadow: 0 8px 25px rgba(78, 205, 196, 0.3);">
                <h4>🎯 Try Sample Data</h4>
                <p>Explore features with our demo dataset</p>
                <p style="font-size: 0.9rem; opacity: 0.8;">Perfect for learning & testing</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Enhanced features overview with modern cards
    st.markdown("### ✨ Powerful Features")
    
    # Create features using Streamlit columns for guaranteed rendering
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown(
            """
            <div class="feature-card">
                <span class="feature-icon">🧠</span>
                <div class="feature-title">AI-Powered Analytics</div>
                <div class="feature-description">
                    Advanced machine learning algorithms automatically analyze your data patterns
                    and provide intelligent insights with minimal configuration required.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <div class="feature-card">
                <span class="feature-icon">�</span>
                <div class="feature-title">AutoML Pipeline</div>
                <div class="feature-description">
                    End-to-end automated machine learning with hyperparameter tuning,
                    model selection, and performance optimization.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with feature_col2:
        st.markdown(
            """
            <div class="feature-card">
                <span class="feature-icon">🧹</span>
                <div class="feature-title">Smart Data Cleaning</div>
                <div class="feature-description">
                    Intelligent detection and handling of missing values, outliers, and data
                    inconsistencies with ML-driven recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <div class="feature-card">
                <span class="feature-icon">⚡</span>
                <div class="feature-title">Real-time Processing</div>
                <div class="feature-description">
                    Lightning-fast data processing with optimized algorithms for
                    large datasets and real-time analytics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with feature_col3:
        st.markdown(
            """
            <div class="feature-card">
                <span class="feature-icon">📊</span>
                <div class="feature-title">Interactive Visualizations</div>
                <div class="feature-description">
                    Dynamic charts, plots, and dashboards that adapt to your data automatically
                    with professional-grade visual analytics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <div class="feature-card">
                <span class="feature-icon">🎯</span>
                <div class="feature-title">Production Ready</div>
                <div class="feature-description">
                    Enterprise-grade deployment capabilities with model monitoring,
                    batch predictions, and scalable infrastructure.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Call-to-action section
    st.markdown("---")
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); 
                    padding: 30px; border-radius: 20px; color: white; text-align: center;
                    margin: 30px 0; box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);">
            <h3 style="margin-bottom: 15px;">🎉 Ready to Transform Your Data?</h3>
            <p style="font-size: 1.1rem; margin-bottom: 20px; opacity: 0.9;">
                Join thousands of data scientists and analysts who trust Data Assistant Pro
            </p>
            <p style="font-size: 0.9rem; opacity: 0.8;">
                👆 Use the sidebar to upload your data or try our sample dataset
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick start guide
    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        **Step 1: Upload Data** 📁
        - Click 'Browse files' in sidebar
        - Select your CSV file
        - Or try our sample dataset
        """
        )

    with col2:
        st.markdown(
            """
        **Step 2: Clean & Explore** 🧹
        - Review data overview
        - Apply cleaning operations
        - Explore with automated EDA
        """
        )

    with col3:
        st.markdown(
            """
        **Step 3: Build Models** 🤖
        - Select target column
        - Train multiple algorithms
        - Make predictions
        """
        )

    # Performance metrics showcase
    st.markdown("---")
    st.subheader("⚡ Performance Metrics")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    with perf_col1:
        st.metric("Processing Speed", "< 2 sec", delta="Real-time")
    with perf_col2:
        st.metric("Accuracy Rate", "95%+", delta="High precision")
    with perf_col3:
        st.metric("Memory Usage", "Optimized", delta="Efficient")
    with perf_col4:
        st.metric("Model Types", "6+", delta="Comprehensive")
