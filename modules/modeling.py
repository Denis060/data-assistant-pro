# In modules/modeling.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def prepare_data_for_modeling(df, target_column, problem_type):
    """Prepare data for machine learning modeling."""
    try:
        # Make a copy of the dataframe
        df_model = df.copy()
        
        # Separate features and target
        X = df_model.drop(columns=[target_column])
        y = df_model[target_column]
        
        # Handle missing values in target
        if y.isnull().any():
            st.warning(f"Target column '{target_column}' has missing values. Removing rows with missing targets.")
            mask = y.notnull()
            X = X[mask]
            y = y[mask]
        
        # Handle categorical variables in features
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            for col in categorical_columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    # Handle missing values in categorical columns
                    X[col] = X[col].fillna('Missing')
                    X[col] = le.fit_transform(X[col])
        
        # Handle missing values in numerical columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            imputer = SimpleImputer(strategy='median')
            X[numerical_columns] = imputer.fit_transform(X[numerical_columns])
        
        # Encode target variable if it's categorical (for classification)
        if problem_type == 'Classification' and y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
        
        return X, y, True
        
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None, None, False

def train_models(X, y, problem_type, test_size=0.2):
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
        
        models = {}
        results = {}
        
        if problem_type == 'Classification':
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42)
            }
            
            for name, model in models.items():
                if name == 'SVM':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'y_test': y_test
                }
        
        else:  # Regression
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'SVM': SVR()
            }
            
            for name, model in models.items():
                if name == 'SVM':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'predictions': y_pred,
                    'y_test': y_test
                }
        
        return results, X_train, X_test, y_train, y_test, scaler
        
    except Exception as e:
        st.error(f"Error training models: {e}")
        return None, None, None, None, None, None

def display_model_results(results, problem_type):
    """Display model results and visualizations."""
    
    if problem_type == 'Classification':
        st.subheader("📊 Classification Results")
        
        # Model comparison
        model_names = []
        accuracies = []
        
        for name, result in results.items():
            model_names.append(name)
            accuracies.append(result['accuracy'])
        
        # Create comparison chart
        fig = px.bar(
            x=model_names, 
            y=accuracies,
            title="Model Accuracy Comparison",
            labels={'x': 'Models', 'y': 'Accuracy'},
            color=accuracies,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model details
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        st.success(f"🏆 Best Model: {best_model[0]} with accuracy: {best_model[1]['accuracy']:.4f}")
        
        # Model metrics table
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': [f"{acc:.4f}" for acc in accuracies]
        })
        st.dataframe(metrics_df, use_container_width=True)
        
    else:  # Regression
        st.subheader("📊 Regression Results")
        
        # Model comparison
        model_names = []
        r2_scores = []
        mse_scores = []
        
        for name, result in results.items():
            model_names.append(name)
            r2_scores.append(result['r2'])
            mse_scores.append(result['mse'])
        
        # Create comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_r2 = px.bar(
                x=model_names, 
                y=r2_scores,
                title="R² Score Comparison",
                labels={'x': 'Models', 'y': 'R² Score'},
                color=r2_scores,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            fig_mse = px.bar(
                x=model_names, 
                y=mse_scores,
                title="Mean Squared Error Comparison",
                labels={'x': 'Models', 'y': 'MSE'},
                color=mse_scores,
                color_continuous_scale='viridis_r'
            )
            st.plotly_chart(fig_mse, use_container_width=True)
        
        # Best model details
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        st.success(f"🏆 Best Model: {best_model[0]} with R² score: {best_model[1]['r2']:.4f}")
        
        # Model metrics table
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'R² Score': [f"{r2:.4f}" for r2 in r2_scores],
            'MSE': [f"{mse:.4f}" for mse in mse_scores]
        })
        st.dataframe(metrics_df, use_container_width=True)

def create_prediction_plots(results, problem_type):
    """Create prediction vs actual plots."""
    
    st.subheader("🎯 Predictions vs Actual Values")
    
    if problem_type == 'Regression':
        # Create subplots for each model
        for name, result in results.items():
            fig = px.scatter(
                x=result['y_test'], 
                y=result['predictions'],
                title=f"{name}: Predictions vs Actual",
                labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                trendline="ols"
            )
            
            # Add perfect prediction line
            min_val = min(min(result['y_test']), min(result['predictions']))
            max_val = max(max(result['y_test']), max(result['predictions']))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            st.plotly_chart(fig, use_container_width=True)

def feature_importance_analysis(results, X, problem_type):
    """Display feature importance for tree-based models."""
    
    st.subheader("🔍 Feature Importance Analysis")
    
    # Get Random Forest model (if available)
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        
        if hasattr(rf_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df.head(10),  # Top 10 features
                x='Importance', 
                y='Feature',
                orientation='h',
                title="Top 10 Most Important Features (Random Forest)",
                labels={'Importance': 'Feature Importance', 'Feature': 'Features'}
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Show importance table
            st.dataframe(importance_df, use_container_width=True)
