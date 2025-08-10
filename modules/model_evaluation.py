"""
Enhanced Model Evaluation and Performance Analysis
Comprehensive tools for model performance assessment and improvement suggestions
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, 
    mean_absolute_error, r2_score, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ModelPerformanceAnalyzer:
    """Comprehensive model performance analysis and improvement suggestions."""
    
    def __init__(self):
        self.performance_history = []
        self.improvement_suggestions = []
    
    def analyze_predictions(self, y_true, y_pred, y_pred_proba=None, 
                          problem_type='classification', feature_names=None,
                          model_name='Model'):
        """
        Comprehensive analysis of model predictions vs actual values.
        
        Args:
            y_true: Actual target values
            y_pred: Predicted values
            y_pred_proba: Prediction probabilities (for classification)
            problem_type: 'classification' or 'regression'
            feature_names: List of feature names
            model_name: Name of the model
        
        Returns:
            Dict with comprehensive analysis results
        """
        analysis_results = {
            'model_name': model_name,
            'problem_type': problem_type,
            'total_predictions': len(y_true),
            'performance_metrics': {},
            'confusion_analysis': {},
            'improvement_suggestions': [],
            'data_quality_issues': []
        }
        
        try:
            # Data validation and preprocessing
            y_true_processed, y_pred_processed = self._validate_and_process_data(y_true, y_pred)
            
            if problem_type == 'classification':
                analysis_results.update(self._analyze_classification(
                    y_true_processed, y_pred_processed, y_pred_proba, model_name
                ))
            else:
                analysis_results.update(self._analyze_regression(
                    y_true_processed, y_pred_processed, model_name
                ))
            
            # Generate improvement suggestions
            analysis_results['improvement_suggestions'] = self._generate_improvement_suggestions(
                analysis_results, y_true_processed, y_pred_processed
            )
            
            # Detect data quality issues
            analysis_results['data_quality_issues'] = self._detect_data_quality_issues(
                y_true_processed, y_pred_processed, feature_names
            )
            
        except Exception as e:
            st.error(f"Error in performance analysis: {str(e)}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def _validate_and_process_data(self, y_true, y_pred):
        """Validate and process input data to ensure compatibility."""
        # Convert to numpy arrays for consistency
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Handle string/object dtypes that might contain dates or categorical values
        if y_true.dtype == 'object':
            try:
                # Try to convert to numeric
                y_true = pd.to_numeric(y_true, errors='coerce')
                # Fill any NaN values created during conversion
                if np.isnan(y_true).any():
                    y_true = np.nan_to_num(y_true, nan=0.0)
            except Exception as e:
                raise ValueError(f"Cannot convert y_true to numeric format. Contains: {np.unique(y_true[:5])}")
        
        if y_pred.dtype == 'object':
            try:
                # Try to convert to numeric
                y_pred = pd.to_numeric(y_pred, errors='coerce')
                # Fill any NaN values created during conversion
                if np.isnan(y_pred).any():
                    y_pred = np.nan_to_num(y_pred, nan=0.0)
            except Exception as e:
                raise ValueError(f"Cannot convert y_pred to numeric format. Contains: {np.unique(y_pred[:5])}")
        
        # Ensure arrays have same length
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)} samples")
        
        # Remove any infinite values
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not finite_mask.all():
            st.warning(f"⚠️ Removed {(~finite_mask).sum()} samples with infinite values")
            y_true = y_true[finite_mask]
            y_pred = y_pred[finite_mask]
        
        return y_true, y_pred
    
    def _analyze_classification(self, y_true, y_pred, y_pred_proba=None, model_name='Model'):
        """Detailed classification analysis."""
        results = {}
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle multiclass vs binary
        average_method = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
        
        precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)
        
        results['performance_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }
        
        # Add AUC if probabilities available and binary classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                results['performance_metrics']['auc_roc'] = auc
            except:
                pass
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # Class-wise performance
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        results['class_wise_performance'] = class_report
        
        # Performance assessment
        results['performance_assessment'] = self._assess_classification_performance(
            accuracy, precision, recall, f1
        )
        
        return results
    
    def _analyze_regression(self, y_true, y_pred, model_name='Model'):
        """Detailed regression analysis."""
        results = {}
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        
        results['performance_metrics'] = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'mape': mape
        }
        
        # Residual analysis
        residuals = y_true - y_pred
        results['residual_analysis'] = {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': float(pd.Series(residuals).skew()),
            'residual_kurtosis': float(pd.Series(residuals).kurtosis())
        }
        
        # Performance assessment
        results['performance_assessment'] = self._assess_regression_performance(r2, rmse, mape)
        
        return results
    
    def _assess_classification_performance(self, accuracy, precision, recall, f1):
        """Assess classification model performance and categorize."""
        
        if accuracy >= 0.9 and precision >= 0.9 and recall >= 0.9:
            return {
                'level': 'Excellent',
                'color': 'green',
                'description': 'Outstanding performance across all metrics',
                'issues': []
            }
        elif accuracy >= 0.8 and precision >= 0.8 and recall >= 0.8:
            issues = []
            if precision < 0.85:
                issues.append('Moderate false positive rate')
            if recall < 0.85:
                issues.append('Moderate false negative rate')
            
            return {
                'level': 'Good',
                'color': 'blue',
                'description': 'Good performance with room for improvement',
                'issues': issues
            }
        elif accuracy >= 0.7:
            issues = []
            if precision < 0.75:
                issues.append('High false positive rate')
            if recall < 0.75:
                issues.append('High false negative rate')
            if accuracy < 0.75:
                issues.append('Low overall accuracy')
            
            return {
                'level': 'Fair',
                'color': 'orange',
                'description': 'Moderate performance, needs improvement',
                'issues': issues
            }
        else:
            return {
                'level': 'Poor',
                'color': 'red',
                'description': 'Poor performance, significant improvements needed',
                'issues': ['Low accuracy', 'Poor precision/recall balance', 'Model may be underfitting']
            }
    
    def _assess_regression_performance(self, r2, rmse, mape):
        """Assess regression model performance."""
        
        if r2 >= 0.9 and mape <= 10:
            return {
                'level': 'Excellent',
                'color': 'green',
                'description': 'Outstanding predictive performance',
                'issues': []
            }
        elif r2 >= 0.8 and mape <= 20:
            issues = []
            if r2 < 0.85:
                issues.append('Some unexplained variance')
            if mape > 15:
                issues.append('Moderate prediction errors')
            
            return {
                'level': 'Good',
                'color': 'blue',
                'description': 'Good predictive performance',
                'issues': issues
            }
        elif r2 >= 0.6:
            issues = []
            if r2 < 0.7:
                issues.append('Significant unexplained variance')
            if mape > 25:
                issues.append('High prediction errors')
            
            return {
                'level': 'Fair',
                'color': 'orange',
                'description': 'Moderate performance, improvements needed',
                'issues': issues
            }
        else:
            return {
                'level': 'Poor',
                'color': 'red',
                'description': 'Poor predictive performance',
                'issues': ['Very low R² score', 'High prediction errors', 'Model may be underfitting']
            }
    
    def _generate_improvement_suggestions(self, analysis_results, y_true, y_pred):
        """Generate specific improvement suggestions based on performance analysis."""
        suggestions = []
        
        problem_type = analysis_results['problem_type']
        metrics = analysis_results['performance_metrics']
        assessment = analysis_results.get('performance_assessment', {})
        
        if problem_type == 'classification':
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            
            # Low accuracy suggestions
            if accuracy < 0.7:
                suggestions.extend([
                    "🔧 **Feature Engineering**: Create polynomial features (age², income×experience), interaction terms (education×salary), or domain-specific features (tenure=current_year-join_year)",
                    "📊 **Data Quality**: Check for data leakage (future info in features), incorrect labels, or systematic errors. Verify target variable consistency",
                    "🤖 **Model Complexity**: Try ensemble methods (Random Forest 100+ trees, XGBoost with 200+ rounds, Voting Classifier combining 3+ algorithms)",
                    "⚖️ **Class Balance**: Address imbalance with SMOTE oversampling, class_weight='balanced' parameter, or stratified sampling with 70-30 split",
                    "🔍 **Hyperparameter Tuning**: Use GridSearchCV with cv=5 folds, or RandomizedSearchCV with 100+ iterations for optimal parameters",
                    "📈 **Feature Selection**: Use SelectKBest with f_classif, RFE with cross-validation, or feature importance from tree-based models",
                    "🎯 **Cross-Validation**: Implement StratifiedKFold with 5-10 folds to get more reliable performance estimates"
                ])
            elif accuracy < 0.85:
                suggestions.extend([
                    "🔧 **Advanced Feature Engineering**: Try logarithmic transforms log(income+1), binning continuous variables, or creating ratio features",
                    "🤖 **Algorithm Optimization**: Fine-tune current model with deeper hyperparameter search or try advanced algorithms like LightGBM, CatBoost",
                    "📊 **Feature Scaling**: Apply StandardScaler for SVM/Neural Networks, MinMaxScaler for distance-based algorithms, or RobustScaler for outlier-resistant scaling"
                ])
            
            # Precision/Recall imbalance
            if abs(precision - recall) > 0.15:
                if precision > recall:
                    suggestions.extend([
                        "🎯 **Improve Recall**: Lower classification threshold from 0.5 to 0.3-0.4, use cost-sensitive learning with class weights, or focus on minority class precision",
                        "⚖️ **Sampling Strategy**: Apply ADASYN adaptive sampling, or combination of SMOTE + Tomek links for better boundary definition"
                    ])
                else:
                    suggestions.extend([
                        "🎯 **Improve Precision**: Increase threshold to 0.6-0.7, add more discriminative features, or use ensemble methods with hard voting",
                        "🔍 **Feature Quality**: Remove noisy features using correlation analysis (>0.95), variance threshold, or mutual information"
                    ])
            
            # Class imbalance detection
            class_counts = pd.Series(y_true).value_counts()
            if len(class_counts) > 1:
                imbalance_ratio = class_counts.max() / class_counts.min()
                if imbalance_ratio > 3:
                    suggestions.append(f"⚖️ **Address Severe Class Imbalance**: Ratio is {imbalance_ratio:.1f}:1. Use SMOTE with k_neighbors=3, class_weight={{0: 1, 1: {imbalance_ratio:.1f}}}, or ensemble with BalancedRandomForest")
        
        else:  # Regression
            r2 = metrics.get('r2_score', 0)
            mape = metrics.get('mape', 100)
            rmse = metrics.get('rmse', float('inf'))
            
            if r2 < 0.7:
                suggestions.extend([
                    "📊 **Advanced Feature Engineering**: Create polynomial features up to degree 2-3, logarithmic transforms log(x+1), square root transforms sqrt(x), or interaction terms between top correlated features",
                    "🔍 **Feature Selection**: Use L1 regularization (Lasso α=0.01-1.0), Recursive Feature Elimination with 10-fold CV, or feature importance from RandomForest with 200+ trees",
                    "🤖 **Advanced Algorithms**: Try Gradient Boosting (XGBoost, LightGBM) with learning_rate=0.1, max_depth=6, or Neural Networks with 2-3 hidden layers",
                    "📈 **Data Preprocessing**: Handle outliers with IQR method (Q1-1.5*IQR, Q3+1.5*IQR), normalize features with StandardScaler, or try robust scaling"
                ])
            elif r2 < 0.85:
                suggestions.extend([
                    "🔧 **Fine-tune Current Model**: Optimize hyperparameters with Bayesian optimization, increase model complexity gradually, or try ensemble stacking",
                    "📊 **Feature Transformation**: Apply Box-Cox or Yeo-Johnson power transforms, create lagged features for time-series, or use PCA for dimensionality reduction"
                ])
            
            if mape > 25:
                suggestions.extend([
                    "🎯 **Reduce High Prediction Errors**: Focus on outlier detection with Isolation Forest, robust preprocessing with RobustScaler, or winsorization at 5th/95th percentiles",
                    "📊 **Target Transformation**: Try log(y+1) transformation, Square root transformation, or Box-Cox with optimal lambda parameter",
                    "� **Robust Cross-Validation**: Use TimeSeriesSplit for temporal data, or RepeatedKFold with 5 folds × 3 repeats for more stable estimates"
                ])
            elif mape > 15:
                suggestions.extend([
                    "🎯 **Optimize Predictions**: Fine-tune regularization parameters (α=0.001-10), try different loss functions, or ensemble with weighted averaging",
                    "📈 **Feature Engineering**: Create domain-specific ratios, moving averages for time series, or categorical encoding improvements"
                ])
        
        # General suggestions based on assessment level
        assessment_level = assessment.get('level', 'Unknown')
        if assessment_level == 'Poor':
            suggestions.extend([
                "📚 **Data Collection**: Gather more training data (aim for 1000+ samples per class), improve data quality with better collection methods",
                "🧹 **Comprehensive Data Cleaning**: Handle missing values with advanced imputation (KNN, iterative), remove or cap extreme outliers (>3 std deviations)",
                "🔄 **Feature Scaling**: Apply appropriate scaling (StandardScaler for normal distribution, MinMaxScaler for bounded features, RobustScaler for outliers)",
                "🎲 **Algorithm Exploration**: Try multiple algorithms (SVM with RBF kernel, Neural Networks with dropout, Ensemble methods with different base learners)",
                "📏 **Robust Validation**: Use 10-fold cross-validation, holdout test set (20%), and temporal validation for time-series data"
            ])
        elif assessment_level == 'Fair':
            suggestions.extend([
                "🔧 **Model Optimization**: Fine-tune hyperparameters with GridSearch (depth=3-10, estimators=50-500), try different algorithms or ensemble methods",
                "📊 **Feature Engineering**: Create interaction terms, polynomial features, or domain-specific transformations based on business logic",
                "🎯 **Performance Boosting**: Use ensemble methods (Voting, Bagging, Boosting) or model stacking with diverse base learners"
            ])
        
        # Add specific code examples for implementation
        suggestions.append("💻 **Implementation Examples**:")
        if problem_type == 'classification':
            suggestions.extend([
                "```python\n# Class balancing with SMOTE\nfrom imblearn.over_sampling import SMOTE\nsmote = SMOTE(k_neighbors=3)\nX_balanced, y_balanced = smote.fit_resample(X_train, y_train)\n```",
                "```python\n# Ensemble with class weights\nfrom sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)\n```"
            ])
        else:
            suggestions.extend([
                "```python\n# Feature engineering for regression\nimport numpy as np\nX['feature_squared'] = X['feature'] ** 2\nX['log_feature'] = np.log1p(X['feature'])\nX['interaction'] = X['feature1'] * X['feature2']\n```",
                "```python\n# Advanced regression with XGBoost\nimport xgboost as xgb\nmodel = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6)\n```"
            ])
        
        return suggestions
    
    def _detect_data_quality_issues(self, y_true, y_pred, feature_names=None):
        """Detect potential data quality issues affecting model performance."""
        issues = []
        
        # Check for prediction patterns
        pred_unique = len(np.unique(y_pred))
        true_unique = len(np.unique(y_true))
        
        if pred_unique < true_unique * 0.5:
            issues.append({
                'type': 'Limited Prediction Range',
                'description': f'Model only predicts {pred_unique} unique values vs {true_unique} in actual data',
                'severity': 'High',
                'suggestion': 'Check for model underfitting or insufficient feature diversity'
            })
        
        # Check for constant predictions
        if pred_unique <= 2 and true_unique > 2:
            issues.append({
                'type': 'Nearly Constant Predictions',
                'description': 'Model produces very limited range of predictions',
                'severity': 'Critical',
                'suggestion': 'Model may be severely underfitting - review features and model complexity'
            })
        
        # Check prediction distribution
        pred_series = pd.Series(y_pred)
        if pred_series.std() == 0:
            issues.append({
                'type': 'Constant Predictions',
                'description': 'Model always predicts the same value',
                'severity': 'Critical',
                'suggestion': 'Check training process, feature quality, and model implementation'
            })
        
        # Check for extreme values
        if len(y_pred) > 0:
            pred_range = np.max(y_pred) - np.min(y_pred)
            true_range = np.max(y_true) - np.min(y_true)
            
            if pred_range > 0 and true_range > 0:
                range_ratio = pred_range / true_range
                if range_ratio < 0.1:
                    issues.append({
                        'type': 'Narrow Prediction Range',
                        'description': f'Prediction range is only {range_ratio:.2%} of actual data range',
                        'severity': 'Medium',
                        'suggestion': 'Model may be overly conservative - check regularization and feature scaling'
                    })
        
        return issues


def create_performance_analysis_dashboard(y_true, y_pred, y_pred_proba=None, 
                                        problem_type='classification', model_name='Model'):
    """Create comprehensive performance analysis dashboard."""
    
    try:
        analyzer = ModelPerformanceAnalyzer()
        analysis = analyzer.analyze_predictions(
            y_true, y_pred, y_pred_proba, problem_type, model_name=model_name
        )
        
        # Check if there was an error in analysis
        if 'error' in analysis:
            st.error(f"❌ **Analysis Error**: {analysis['error']}")
            st.info("💡 **Common fixes:**")
            st.write("• Check that your target variable contains only numeric values")
            st.write("• Ensure predictions and actual values have the same format")
            st.write("• Verify that categorical values were properly encoded")
            return None
        
        st.markdown("## 📊 Comprehensive Model Performance Analysis")
        
        # Performance Overview
        assessment = analysis.get('performance_assessment', {})
        level = assessment.get('level', 'Unknown')
        color = assessment.get('color', 'gray')
        description = assessment.get('description', 'No assessment available')
        
        # Performance level indicator
        if level == 'Excellent':
            st.success(f"🏆 **{level} Performance**: {description}")
        elif level == 'Good':
            st.info(f"✅ **{level} Performance**: {description}")
        elif level == 'Fair':
            st.warning(f"⚠️ **{level} Performance**: {description}")
        else:
            st.error(f"❌ **{level} Performance**: {description}")
        
        # Key Metrics
        st.markdown("### 📈 Key Performance Metrics")
        metrics = analysis['performance_metrics']
        
        if problem_type == 'classification':
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            with col2:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            with col3:
                st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            with col4:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
            
            # Confusion Matrix
            if 'confusion_matrix' in analysis:
                st.markdown("### 🔍 Confusion Matrix Analysis")
                cm = analysis['confusion_matrix']
                
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    title="Confusion Matrix",
                    labels=dict(x="Predicted", y="Actual", color="Count")
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:  # Regression
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score", f"{metrics.get('r2_score', 0):.3f}")
            with col2:
                st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
            with col3:
                st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
            with col4:
                st.metric("MAPE", f"{metrics.get('mape', 0):.1f}%")
            
            # Prediction vs Actual plot - with error handling for date/string values
            st.markdown("### 📊 Predictions vs Actual Values")
            try:
                fig = px.scatter(
                    x=y_true, y=y_pred,
                    title="Predictions vs Actual Values",
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'}
                )
                
                # Add perfect prediction line
                min_val = min(min(y_true), min(y_pred))
                max_val = max(max(y_true), max(y_pred))
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Predictions',
                    line=dict(dash='dash', color='red')
                ))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as plot_error:
                st.error(f"❌ Could not create prediction plot: {str(plot_error)}")
                st.info("💡 This often happens when data contains non-numeric values")
        
        # Issues and Suggestions
        issues = assessment.get('issues', [])
        if issues:
            st.markdown("### ⚠️ Identified Issues")
            for issue in issues:
                st.warning(f"• {issue}")
        
        # Data Quality Issues
        data_issues = analysis.get('data_quality_issues', [])
        if data_issues:
            st.markdown("### 🔍 Data Quality Issues")
            for issue in data_issues:
                severity = issue.get('severity', 'Medium')
                if severity == 'Critical':
                    st.error(f"**{issue['type']}**: {issue['description']}")
                    st.error(f"💡 Suggestion: {issue['suggestion']}")
                elif severity == 'High':
                    st.warning(f"**{issue['type']}**: {issue['description']}")
                    st.info(f"💡 Suggestion: {issue['suggestion']}")
                else:
                    st.info(f"**{issue['type']}**: {issue['description']}")
                    st.info(f"💡 Suggestion: {issue['suggestion']}")
        
        # Improvement Suggestions
        suggestions = analysis.get('improvement_suggestions', [])
        if suggestions:
            st.markdown("### 🚀 Improvement Suggestions")
            st.markdown("Based on the performance analysis, here are specific recommendations:")
            
            for i, suggestion in enumerate(suggestions, 1):
                st.markdown(f"{i}. {suggestion}")
        
        # Download detailed report
        if st.button("📥 Download Detailed Analysis Report"):
            report = generate_performance_report(analysis, y_true, y_pred)
            st.download_button(
                "📄 Download Report",
                data=report,
                file_name=f"performance_analysis_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        return analysis
    
    except Exception as e:
        st.error(f"❌ **Dashboard Error**: {str(e)}")
        st.info("💡 **Possible solutions:**")
        st.write("• Check that your input data contains only numeric values")
        st.write("• Ensure target variable is properly formatted")
        st.write("• Try preprocessing your data before analysis")
        
        # Show debugging information
        with st.expander("🔍 Debug Information"):
            st.write("**Error Details:**", str(e))
            st.write("**Error Type:**", type(e).__name__)
            st.write("**Model Name:**", model_name)
            st.write("**Problem Type:**", problem_type)
            if hasattr(y_true, 'dtype'):
                st.write("**y_true dtype:**", str(y_true.dtype))
            if hasattr(y_pred, 'dtype'):
                st.write("**y_pred dtype:**", str(y_pred.dtype))
        
        return None
    if st.button("📥 Download Detailed Analysis Report"):
        report = generate_performance_report(analysis, y_true, y_pred)
        st.download_button(
            "📄 Download Report",
            data=report,
            file_name=f"performance_analysis_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    return analysis


def generate_performance_report(analysis, y_true, y_pred):
    """Generate detailed text report of performance analysis."""
    
    report = f"""
MODEL PERFORMANCE ANALYSIS REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {analysis['model_name']}
Problem Type: {analysis['problem_type'].title()}

PERFORMANCE OVERVIEW:
{'-' * 50}
Assessment Level: {analysis.get('performance_assessment', {}).get('level', 'Unknown')}
Description: {analysis.get('performance_assessment', {}).get('description', 'No description')}

PERFORMANCE METRICS:
{'-' * 50}
"""
    
    metrics = analysis['performance_metrics']
    for metric, value in metrics.items():
        if isinstance(value, float):
            report += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"
        else:
            report += f"{metric.replace('_', ' ').title()}: {value}\n"
    
    # Issues
    issues = analysis.get('performance_assessment', {}).get('issues', [])
    if issues:
        report += f"\nIDENTIFIED ISSUES:\n{'-' * 50}\n"
        for issue in issues:
            report += f"• {issue}\n"
    
    # Data Quality Issues
    data_issues = analysis.get('data_quality_issues', [])
    if data_issues:
        report += f"\nDATA QUALITY ISSUES:\n{'-' * 50}\n"
        for issue in data_issues:
            report += f"• {issue['type']}: {issue['description']}\n"
            report += f"  Severity: {issue['severity']}\n"
            report += f"  Suggestion: {issue['suggestion']}\n\n"
    
    # Improvement Suggestions
    suggestions = analysis.get('improvement_suggestions', [])
    if suggestions:
        report += f"\nIMPROVEMENT SUGGESTIONS:\n{'-' * 50}\n"
        for i, suggestion in enumerate(suggestions, 1):
            # Remove markdown formatting for text report
            clean_suggestion = suggestion.replace('**', '').replace('🔧', '').replace('📊', '').replace('🤖', '').replace('⚖️', '').replace('🔍', '').replace('🎯', '').replace('📚', '').replace('🧹', '').replace('🔄', '').replace('🎲', '').replace('📏', '').replace('📈', '').replace('🚀', '')
            report += f"{i}. {clean_suggestion}\n"
    
    report += f"\nANALYSIS SUMMARY:\n{'-' * 50}\n"
    report += f"Total Predictions: {analysis['total_predictions']}\n"
    report += f"Analysis completed successfully.\n"
    
    return report
