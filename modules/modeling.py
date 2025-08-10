# In modules/modeling.py

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from .cache_utils import DataCache, with_progress_cache
from .error_handler import error_handler, display_smart_error, SmartError, ErrorSeverity
import hashlib
import time

# Import diagnostic modules
try:
    from modules.cleaning_fixed import ModelPerformanceDiagnostic, create_performance_improvement_plan
except ImportError:
    # Fallback if diagnostic modules are not available
    ModelPerformanceDiagnostic = None


def get_model_cache_key(X, y, problem_type, test_size, selected_models):
    """Generate a cache key for model training based on data and parameters"""
    try:
        # Create hash based on data shape, problem type, and model selection
        data_info = f"{X.shape}_{y.shape}_{problem_type}_{test_size}_{sorted(selected_models or [])}"
        # Add sample of data for uniqueness
        data_sample = f"{X.iloc[:5].to_string()}_{y.iloc[:5].to_string()}" if hasattr(X, 'iloc') else str(X[:5])
        cache_content = f"{data_info}_{data_sample}"
        return hashlib.md5(cache_content.encode()).hexdigest()
    except Exception:
        return f"model_{int(time.time())}"


@st.cache_data(ttl=7200)  # Cache for 2 hours
def cached_model_training(cache_key: str, X_data, y_data, problem_type, test_size, selected_models):
    """Cached model training function"""
    # Convert back to appropriate format
    X = pd.DataFrame(X_data) if isinstance(X_data, dict) else X_data
    y = pd.Series(y_data) if isinstance(y_data, list) else y_data
    
    # Call the actual training function
    return train_models_internal(X, y, problem_type, test_size, selected_models)
    create_performance_improvement_plan = None
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

# Import performance diagnostic tool
from modules.performance_diagnostic import ModelPerformanceDiagnostic, create_performance_improvement_plan


def prepare_data_for_modeling(df, target_column, problem_type):
    """Prepare data for machine learning modeling."""
    try:
        # Make a copy of the dataframe
        df_model = df.copy()

        # Separate features and target
        X = df_model.drop(columns=[target_column])
        y = df_model[target_column]

        # Identify and exclude ID-like columns
        id_like_columns = []
        for col in X.columns:
            # Check if column name suggests it's an ID
            col_lower = col.lower()
            if any(
                id_keyword in col_lower for id_keyword in ["id", "_id", "uuid", "key"]
            ):
                id_like_columns.append(col)
            # Check if all values are unique (likely an ID)
            elif X[col].nunique() == len(X):
                id_like_columns.append(col)
            # Check if it's a string column with mostly unique values
            elif X[col].dtype == "object" and X[col].nunique() / len(X) > 0.95:
                id_like_columns.append(col)

        if id_like_columns:
            st.warning(f"Excluding ID-like columns from modeling: {id_like_columns}")
            X = X.drop(columns=id_like_columns)

        # Handle missing values in target
        if y.isnull().any():
            st.warning(
                f"Target column '{target_column}' has missing values. Removing rows with missing targets."
            )
            mask = y.notnull()
            X = X[mask]
            y = y[mask]

        # Handle categorical variables in features
        label_encoders = {}
        categorical_columns = X.select_dtypes(include=["object"]).columns
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                if X[col].dtype == "object":
                    le = LabelEncoder()
                    # Handle missing values in categorical columns
                    X[col] = X[col].fillna("Missing")
                    X[col] = le.fit_transform(X[col])
                    label_encoders[col] = le

        # Handle missing values in numerical columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            imputer = SimpleImputer(strategy="median")
            X[numerical_columns] = imputer.fit_transform(X[numerical_columns])

        # Encode target variable if it's categorical (for classification)
        target_encoder = None
        if problem_type == "Classification" and y.dtype == "object":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)

        return X, y, True, label_encoders, target_encoder

    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None, None, False, None, None


def train_models_internal(X, y, problem_type, test_size=0.2, selected_models=None):
    """Train multiple models and return results."""
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features for SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define all available models
        if problem_type == "Classification":
            all_models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "SVM": SVC(random_state=42, probability=True), # Enable probability for ROC-AUC
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB()
            }
        else:  # Regression
            all_models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Linear Regression": LinearRegression(),
                "SVM": SVR(),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "K-Nearest Neighbors": KNeighborsRegressor(),
            }
        
        # Use selected models or default models
        if selected_models:
            models = {name: all_models[name] for name in selected_models if name in all_models}
        else:
            # Default models for backward compatibility
            if problem_type == "Classification":
                models = {
                    "Random Forest": all_models["Random Forest"],
                    "Logistic Regression": all_models["Logistic Regression"],
                    "SVM": all_models["SVM"]
                }
            else:
                models = {
                    "Random Forest": all_models["Random Forest"],
                    "Linear Regression": all_models["Linear Regression"],
                    "SVM": all_models["SVM"]
                }

        results = {}

        if problem_type == "Classification":
            for name, model in models.items():
                if name == "SVM":
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                results[name] = {
                    "model": model,
                    "accuracy": report.get('accuracy', 0),
                    "precision": report.get('weighted avg', {}).get('precision', 0),
                    "recall": report.get('weighted avg', {}).get('recall', 0),
                    "f1_score": report.get('weighted avg', {}).get('f1-score', 0),
                    "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
                    "confusion_matrix": confusion_matrix(y_test, y_pred),
                    "predictions": y_pred,
                    "y_test": y_test,
                }

        else:  # Regression
            for name, model in models.items():
                if name == "SVM":
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                results[name] = {
                    "model": model,
                    "r2_score": r2_score(y_test, y_pred),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "mse": mean_squared_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "predictions": y_pred,
                    "y_test": y_test,
                }

        # Run performance diagnostic
        diagnostic_results = None
        improvement_plan = None
        try:
            trained_models = {name: result["model"] for name, result in results.items()}
            diagnostic = ModelPerformanceDiagnostic(
                X_train, X_test, y_train, y_test, 
                problem_type.lower()
            )
            diagnostic_results = diagnostic.run_full_diagnostic(trained_models)
            improvement_plan = create_performance_improvement_plan(diagnostic_results)
            
        except Exception as e:
            st.warning(f"Performance diagnostic failed: {str(e)}")

        return results, X_train, X_test, y_train, y_test, scaler, diagnostic_results, improvement_plan

    except MemoryError as e:
        context = {
            'data_shape': (X.shape[0], X.shape[1]) if X is not None else (None, None),
            'problem_type': problem_type,
            'selected_models': selected_models,
            'operation': 'model_training'
        }
        
        smart_error = error_handler.analyze_error(e, context)
        action = display_smart_error(smart_error)
        
        if action == "sample_data_50":
            st.info("💡 Try reducing your dataset size in the Data Overview section")
        elif action == "show_column_selector":
            st.info("💡 Consider feature selection to reduce dimensionality")
            
        return None, None, None, None, None, None, None, None
        
    except ValueError as e:
        # Handle convergence and other model-specific errors
        if "did not converge" in str(e) or "max_iter" in str(e):
            context = {
                'problem_type': problem_type,
                'selected_models': selected_models,
                'error_type': 'convergence'
            }
            
            smart_error = error_handler.analyze_error(e, context)
            action = display_smart_error(smart_error)
            
            if action == "increase_max_iter":
                st.info("💡 Try increasing max_iter in the Advanced Model Optimizer section")
            elif action == "scale_features":
                st.info("💡 Apply feature scaling in the Data Cleaning section")
        else:
            # Generic ValueError handling
            context = {
                'problem_type': problem_type,
                'data_shape': (X.shape[0], X.shape[1]) if X is not None else (None, None),
                'operation': 'model_training'
            }
            
            smart_error = error_handler.analyze_error(e, context)
            display_smart_error(smart_error)
            
        return None, None, None, None, None, None, None, None
        
    except Exception as e:
        context = {
            'problem_type': problem_type,
            'data_shape': (X.shape[0], X.shape[1]) if X is not None else (None, None),
            'selected_models': selected_models,
            'operation': 'model_training'
        }
        
        smart_error = error_handler.analyze_error(e, context)
        display_smart_error(smart_error)
        return None, None, None, None, None, None, None, None


def train_models(X, y, problem_type, test_size=0.2, selected_models=None):
    """Enhanced model training function with caching and progress indicators"""
    
    # Generate cache key
    cache_key = get_model_cache_key(X, y, problem_type, test_size, selected_models)
    
    # Convert data for caching (must be serializable)
    X_data = X.to_dict() if hasattr(X, 'to_dict') else X
    y_data = y.tolist() if hasattr(y, 'tolist') else y
    
    # Use cached training with progress indication
    return with_progress_cache(
        f"Training {len(selected_models or ['all'])} model(s)",
        cached_model_training,
        cache_key, X_data, y_data, problem_type, test_size, selected_models
    )


def display_model_results(results, problem_type):
    """Display model results and visualizations."""
    
    model_results = results

    if problem_type == "Classification":
        st.subheader("📊 Classification Results")

        # Model comparison
        model_names = []
        accuracies = []

        for name, result in model_results.items():
            model_names.append(name)
            accuracies.append(result["accuracy"])

        # Create comparison chart
        fig = px.bar(
            x=model_names,
            y=accuracies,
            title="Model Accuracy Comparison",
            labels={"x": "Models", "y": "Accuracy"},
            color=accuracies,
            color_continuous_scale="viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Best model details
        best_model = max(model_results.items(), key=lambda x: x[1]["accuracy"])
        st.success(
            f"🏆 Best Model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}"
        )

        # Model metrics table
        metrics_df = pd.DataFrame(
            {"Model": model_names, "Accuracy": [f"{acc:.4f}" for acc in accuracies]}
        )
        st.dataframe(metrics_df, use_container_width=True)

    else:  # Regression
        st.subheader("📊 Regression Results")

        # Model comparison
        model_names = []
        r2_scores = []
        mae_scores = []
        mse_scores = []
        rmse_scores = []

        for name, result in model_results.items():
            model_names.append(name)
            r2_scores.append(result.get("r2_score", 0))
            mae_scores.append(result.get("mae", 0))
            mse_scores.append(result.get("mse", 0))
            rmse_scores.append(result.get("rmse", 0))

        # Create comparison charts
        col1, col2 = st.columns(2)

        with col1:
            fig_r2 = px.bar(
                x=model_names,
                y=r2_scores,
                title="R² Score Comparison",
                labels={"x": "Models", "y": "R² Score"},
                color=r2_scores,
                color_continuous_scale="viridis",
            )
            st.plotly_chart(fig_r2, use_container_width=True)

        with col2:
            fig_mse = px.bar(
                x=model_names,
                y=mse_scores,
                title="Mean Squared Error Comparison",
                labels={"x": "Models", "y": "MSE"},
                color=mse_scores,
                color_continuous_scale="viridis_r",
            )
            st.plotly_chart(fig_mse, use_container_width=True)

        # Best model details
        best_model_item = max(model_results.items(), key=lambda x: x[1].get("r2_score", -np.inf))
        st.success(
            f"🏆 Best Model: {best_model_item[0]} with R² score: {best_model_item[1].get('r2_score', 0):.4f}"
        )

        # Model metrics table
        metrics_df = pd.DataFrame(
            {
                "Model": model_names,
                "R² Score": [f"{r2:.4f}" for r2 in r2_scores],
                "MAE": [f"{mae:.4f}" for mae in mae_scores],
                "MSE": [f"{mse:.4f}" for mse in mse_scores],
                "RMSE": [f"{rmse:.4f}" for rmse in rmse_scores],
            }
        )
        st.dataframe(metrics_df, use_container_width=True)


def create_prediction_plots(results, problem_type):
    """Create prediction vs actual plots."""

    st.subheader("🎯 Predictions vs Actual Values")
    
    model_results = results

    if problem_type == "Regression":
        # Create subplots for each model
        for name, result in model_results.items():
            try:
                fig = px.scatter(
                    x=result["y_test"],
                    y=result["predictions"],
                    title=f"{name}: Predictions vs Actual",
                    labels={"x": "Actual Values", "y": "Predicted Values"},
                    trendline="ols",
                )
            except Exception as e:
                # If trendline fails (missing statsmodels), create without trendline
                fig = px.scatter(
                    x=result["y_test"],
                    y=result["predictions"],
                    title=f"{name}: Predictions vs Actual",
                    labels={"x": "Actual Values", "y": "Predicted Values"},
                )

            # Add perfect prediction line
            min_val = min(min(result["y_test"]), min(result["predictions"]))
            max_val = max(max(result["y_test"]), max(result["predictions"]))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Perfect Prediction",
                    line=dict(dash="dash", color="red"),
                )
            )

            st.plotly_chart(fig, use_container_width=True)


def feature_importance_analysis(results, X, problem_type):
    """Display feature importance for tree-based models."""

    st.subheader("🔍 Feature Importance Analysis")
    
    model_results = results

    # Get Random Forest model (if available)
    if "Random Forest" in model_results:
        rf_model = model_results["Random Forest"]["model"]

        if hasattr(rf_model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {"Feature": X.columns, "Importance": rf_model.feature_importances_}
            ).sort_values("Importance", ascending=False)

            fig = px.bar(
                importance_df.head(10),  # Top 10 features
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 10 Most Important Features (Random Forest)",
                labels={"Importance": "Feature Importance", "Feature": "Features"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

            # Show importance table
            st.dataframe(importance_df, use_container_width=True)


def display_performance_diagnostic(diagnostic_results, improvement_plan):
    """Display comprehensive performance diagnostic and improvement suggestions without using expanders."""
    
    if not diagnostic_results or not improvement_plan:
        return
    
    st.markdown("---")
    st.subheader("🔍 **Performance Diagnostic & Improvement Plan**")

    # Custom CSS for styled boxes
    st.markdown("""
    <style>
    .diag-box {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid;
    }
    .problem-box {
        border-color: #FF4B4B;
        background-color: rgba(255, 75, 75, 0.1);
    }
    .impact-box {
        border-color: #FFC700;
        background-color: rgba(255, 199, 0, 0.1);
    }
    .solution-box {
        border-color: #28A745;
        background-color: rgba(40, 167, 69, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Overview
    total_issues = len(diagnostic_results['overall_recommendations'])
    critical_issues = len(diagnostic_results['priority_fixes'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Issues Found", total_issues)
    with col2:
        st.metric("Critical Issues", critical_issues)
    with col3:
        st.metric("Expected Improvement", f"{improvement_plan['expected_improvement']:.0f}%")
    
    # Priority Fixes
    if diagnostic_results['priority_fixes']:
        st.subheader("🚨 **Immediate Action Required (High Priority)**")
        for i, issue in enumerate(diagnostic_results['priority_fixes'], 1):
            st.markdown(f"#### 🔴 **Critical Issue #{i}: {issue['type']}**")
            st.markdown(f'<div class="diag-box problem-box"><b>Problem:</b> {issue["description"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="diag-box impact-box"><b>Impact:</b> {issue["impact"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="diag-box solution-box"><b>Solution:</b> {issue["solution"]}</div>', unsafe_allow_html=True)
            st.markdown("**Implementation:**")
            st.code(issue['fix_code'], language='python')
            if 'affected_features' in issue and issue['affected_features']:
                st.info(f"**Affected Features:** {', '.join(issue['affected_features'][:5])}{'...' if len(issue['affected_features']) > 5 else ''}")
            if 'affected_models' in issue and issue['affected_models']:
                st.info(f"**Affected Models:** {', '.join(issue['affected_models'])}")
    
    # Improvement Plan
    st.subheader("📋 **Step-by-Step Improvement Plan**")
    
    # Immediate fixes
    if improvement_plan['immediate_fixes']:
        st.markdown("#### 🔴 **Step 1: Immediate Fixes (High Impact, Low Effort)**")
        st.markdown("**Expected Improvement: 20-50% better performance**")
        for fix in improvement_plan['immediate_fixes']:
            st.markdown(f"• **Action:** {fix['action']}")
            st.code(fix['code'], language='python')
            st.caption(f"Expected gain: {fix['expected_gain']} | Effort: {fix['effort']}")
            st.markdown("---")
    
    # Short-term improvements
    if improvement_plan['short_term_improvements']:
        st.markdown("#### 🟡 **Step 2: Short-term Improvements (Medium Impact)**")
        st.markdown("**Expected Improvement: 10-30% better performance**")
        for fix in improvement_plan['short_term_improvements']:
            st.markdown(f"• **Action:** {fix['action']}")
            st.code(fix['code'], language='python')
            st.caption(f"Expected gain: {fix['expected_gain']} | Effort: {fix['effort']}")
            st.markdown("---")

    # Long-term strategies
    if improvement_plan['long_term_strategies']:
        st.markdown("#### 🔵 **Step 3: Long-term Strategies (Optimization)**")
        st.markdown("**Expected Improvement: 5-15% better performance**")
        for fix in improvement_plan['long_term_strategies']:
            st.markdown(f"• **Action:** {fix['action']}")
            st.code(fix['code'], language='python')
            st.caption(f"Expected gain: {fix['expected_gain']} | Effort: {fix['effort']}")
            st.markdown("---")

    # Detailed Analysis
    st.subheader("📊 **Detailed Diagnostic Analysis**")
    col_diag1, col_diag2 = st.columns(2)
    with col_diag1:
        st.markdown("**Data Quality Issues:**")
        if diagnostic_results['data_issues']:
            for issue in diagnostic_results['data_issues']:
                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🔵"}
                st.markdown(f"{severity_color.get(issue['severity'], '⚪')} {issue['type']}: {issue['description']}")
        else:
            st.write("No data quality issues found.")
        
        st.markdown("**Model Issues:**")
        if diagnostic_results['model_issues']:
            for issue in diagnostic_results['model_issues']:
                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🔵"}
                st.markdown(f"{severity_color.get(issue['severity'], '⚪')} {issue['type']}: {issue['description']}")
        else:
            st.write("No model issues found.")

    with col_diag2:
        st.markdown("**Feature Issues:**")
        if diagnostic_results['feature_issues']:
            for issue in diagnostic_results['feature_issues']:
                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🔵"}
                st.markdown(f"{severity_color.get(issue['severity'], '⚪')} {issue['type']}: {issue['description']}")
        else:
            st.write("No feature issues found.")

        st.markdown("**Training Issues:**")
        if diagnostic_results['training_issues']:
            for issue in diagnostic_results['training_issues']:
                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🔵"}
                st.markdown(f"{severity_color.get(issue['severity'], '⚪')} {issue['type']}: {issue['description']}")
        else:
            st.write("No training issues found.")
    
    # Summary recommendations
    st.markdown("### 💡 **Quick Summary & Next Steps**")
    
    if critical_issues > 0:
        st.error(f"⚠️ **Urgent:** You have {critical_issues} critical issues that are significantly hurting your model performance. Fix these first!")
    elif total_issues > 0:
        st.warning(f"📈 **Optimization Opportunity:** {total_issues} issues found that could improve your model performance.")
    else:
        st.success("🎉 **Great Job!** Your model setup looks good. Consider hyperparameter tuning for further improvements.")
    
    # Top 3 recommendations
    if diagnostic_results['overall_recommendations']:
        st.markdown("**Top 3 Recommendations:**")
        for i, rec in enumerate(diagnostic_results['overall_recommendations'][:3], 1):
            st.markdown(f"{i}. **{rec['type']}**: {rec['solution']}")
    
    return diagnostic_results, improvement_plan
