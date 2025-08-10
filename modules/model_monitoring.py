"""
Model Performance Monitoring and Drift Detection
Track model performance over time and detect data/concept drift
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
import os

# Utility function to make dataframes Arrow-compatible
def make_arrow_compatible(df):
    """Convert DataFrame columns to Arrow-compatible types."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == object:
            # Check if it contains dtype objects or other problematic types
            try:
                if hasattr(df_copy[col].iloc[0], 'name') if len(df_copy) > 0 else False:  # pandas dtype object
                    df_copy[col] = df_copy[col].astype(str)
            except (IndexError, AttributeError):
                pass
    return df_copy

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(self):
        self.performance_history = []
        self.drift_threshold = 0.1
        self.configuration = {
            'drift_threshold': 0.1,
            'monitoring_enabled': True,
            'alert_threshold': 0.05,
            'min_samples_for_drift': 10,
            'performance_window': 100
        }
        
    def log_prediction(self, 
                      model_name: str,
                      prediction: Any,
                      actual: Optional[Any] = None,
                      features: Optional[Dict] = None,
                      timestamp: Optional[datetime] = None):
        """Log a single prediction for monitoring."""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        log_entry = {
            'timestamp': timestamp,
            'model_name': model_name,
            'prediction': prediction,
            'actual': actual,
            'features': features or {},
            'correct': actual == prediction if actual is not None else None
        }
        
        self.performance_history.append(log_entry)
        
    def calculate_performance_metrics(self, 
                                    model_name: str,
                                    time_window: timedelta = timedelta(days=7)) -> Dict[str, float]:
        """Calculate performance metrics for a time window."""
        
        cutoff_time = datetime.now() - time_window
        recent_predictions = [
            entry for entry in self.performance_history
            if (entry['model_name'] == model_name and 
                entry['timestamp'] >= cutoff_time and
                entry['actual'] is not None)
        ]
        
        if not recent_predictions:
            return {}
            
        # Calculate accuracy
        correct_predictions = sum(1 for entry in recent_predictions if entry['correct'])
        accuracy = correct_predictions / len(recent_predictions)
        
        # Calculate prediction distribution
        predictions = [entry['prediction'] for entry in recent_predictions]
        unique_predictions = len(set(predictions))
        
        return {
            'accuracy': accuracy,
            'total_predictions': len(recent_predictions),
            'unique_predictions': unique_predictions,
            'prediction_rate': len(recent_predictions) / time_window.days
        }
    
    def get_performance_history(self) -> List[Dict]:
        """Get the complete performance history."""
        return self.performance_history.copy()
    
    def clear_history(self):
        """Clear all performance history."""
        self.performance_history.clear()
    
    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific model."""
        model_entries = [
            entry for entry in self.performance_history
            if entry['model_name'] == model_name
        ]
        
        if not model_entries:
            return {'model_name': model_name, 'total_predictions': 0}
        
        # Calculate basic stats
        total_predictions = len(model_entries)
        predictions_with_actual = [e for e in model_entries if e['actual'] is not None]
        
        accuracy = None
        if predictions_with_actual:
            correct = sum(1 for e in predictions_with_actual if e['correct'])
            accuracy = correct / len(predictions_with_actual)
        
        # Time range
        timestamps = [e['timestamp'] for e in model_entries]
        first_prediction = min(timestamps)
        last_prediction = max(timestamps)
        
        return {
            'model_name': model_name,
            'total_predictions': total_predictions,
            'predictions_with_ground_truth': len(predictions_with_actual),
            'accuracy': accuracy,
            'first_prediction': first_prediction,
            'last_prediction': last_prediction,
            'active_period_days': (last_prediction - first_prediction).days + 1
        }
    
    def track_performance(self, performance_data: Dict[str, Any]):
        """Track model performance data."""
        # Add timestamp if not present
        if 'timestamp' not in performance_data:
            performance_data['timestamp'] = datetime.now()
        
        # Add to performance history
        self.performance_history.append(performance_data)
        
        # Log the tracking
        logger.info(f"Tracked performance for model: {performance_data.get('model_name', 'Unknown')}")
    
    def get_model_names(self) -> List[str]:
        """Get list of all tracked model names."""
        model_names = set()
        for entry in self.performance_history:
            if 'model_name' in entry:
                model_names.add(entry['model_name'])
        return list(model_names)
    
    def detect_data_drift(self, 
                         current_data: pd.DataFrame,
                         reference_data: pd.DataFrame,
                         method: str = 'statistical') -> Dict[str, Any]:
        """Detect data drift between current and reference datasets."""
        
        drift_results = {}
        
        for column in current_data.columns:
            if column in reference_data.columns:
                if pd.api.types.is_numeric_dtype(current_data[column]):
                    # Numerical drift detection
                    drift_score = self._detect_numerical_drift(
                        current_data[column], 
                        reference_data[column]
                    )
                else:
                    # Categorical drift detection
                    drift_score = self._detect_categorical_drift(
                        current_data[column], 
                        reference_data[column]
                    )
                
                drift_results[column] = {
                    'drift_score': drift_score,
                    'drift_detected': drift_score > self.drift_threshold,
                    'severity': self._get_drift_severity(drift_score)
                }
        
        return drift_results
    
    def _detect_numerical_drift(self, current: pd.Series, reference: pd.Series) -> float:
        """Detect drift in numerical columns using Kolmogorov-Smirnov test."""
        try:
            from scipy import stats
            statistic, p_value = stats.ks_2samp(current.dropna(), reference.dropna())
            return 1 - p_value  # Higher score means more drift
        except ImportError:
            # Fallback to simple statistical comparison
            current_stats = current.describe()
            reference_stats = reference.describe()
            
            mean_diff = abs(current_stats['mean'] - reference_stats['mean']) / reference_stats['std']
            std_diff = abs(current_stats['std'] - reference_stats['std']) / reference_stats['std']
            
            return min(1.0, (mean_diff + std_diff) / 2)
    
    def _detect_categorical_drift(self, current: pd.Series, reference: pd.Series) -> float:
        """Detect drift in categorical columns using distribution comparison."""
        current_dist = current.value_counts(normalize=True)
        reference_dist = reference.value_counts(normalize=True)
        
        # Calculate JS divergence as drift measure
        all_categories = set(current_dist.index) | set(reference_dist.index)
        
        current_probs = [current_dist.get(cat, 0) for cat in all_categories]
        reference_probs = [reference_dist.get(cat, 0) for cat in all_categories]
        
        # Simple drift measure based on probability differences
        drift_score = sum(abs(c - r) for c, r in zip(current_probs, reference_probs)) / 2
        
        return drift_score
    
    def _get_drift_severity(self, drift_score: float) -> str:
        """Get drift severity level."""
        if drift_score < 0.1:
            return "Low"
        elif drift_score < 0.3:
            return "Medium"
        else:
            return "High"

    def create_comparison_dashboard(self, performance_history: List[Dict]) -> Dict[str, Any]:
        """Create a comparison dashboard for multiple models."""
        try:
            if not performance_history or len(performance_history) < 2:
                return None
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(performance_history)
            
            # Group by model
            model_groups = df.groupby('model_name')
            
            comparison_metrics = []
            model_rankings = []
            
            for model_name, group in model_groups:
                if len(group) == 0:
                    continue
                    
                # Calculate metrics for this model
                metrics = self.calculate_performance_metrics(model_name)
                
                if metrics:
                    comparison_metrics.append({
                        'Model': model_name,
                        'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                        'Predictions': len(group),
                        'Latest_Performance': f"{group['correct'].tail(10).mean():.3f}" if 'correct' in group.columns and not group['correct'].isna().all() else "N/A",
                        'Avg_Performance': f"{group['correct'].mean():.3f}" if 'correct' in group.columns and not group['correct'].isna().all() else "N/A"
                    })
                    
                    model_rankings.append({
                        'model': model_name,
                        'score': metrics.get('accuracy', 0),
                        'predictions': len(group)
                    })
            
            # Sort rankings by performance
            model_rankings.sort(key=lambda x: x['score'], reverse=True)
            
            # Create visualizations
            comparison_charts = {}
            
            if len(comparison_metrics) > 1:
                # Performance comparison chart
                metrics_df = pd.DataFrame(comparison_metrics)
                
                if not metrics_df.empty and 'Accuracy' in metrics_df.columns:
                    # Convert accuracy back to float for plotting
                    try:
                        metrics_df['Accuracy_Float'] = metrics_df['Accuracy'].astype(float)
                        
                        fig = px.bar(
                            metrics_df, 
                            x='Model', 
                            y='Accuracy_Float',
                            title='Model Performance Comparison',
                            labels={'Accuracy_Float': 'Accuracy'}
                        )
                        fig.update_layout(showlegend=False)
                        comparison_charts['performance_comparison'] = fig
                    except:
                        pass
                
                # Performance over time
                if 'timestamp' in df.columns and 'correct' in df.columns:
                    time_comparison = []
                    for model_name, group in model_groups:
                        if 'correct' in group.columns and not group['correct'].isna().all():
                            # Calculate rolling accuracy
                            group_sorted = group.sort_values('timestamp')
                            rolling_acc = group_sorted['correct'].rolling(window=min(5, len(group_sorted)), min_periods=1).mean()
                            
                            for idx, (timestamp, acc) in enumerate(zip(group_sorted['timestamp'], rolling_acc)):
                                time_comparison.append({
                                    'timestamp': timestamp,
                                    'accuracy': acc,
                                    'model': model_name,
                                    'prediction_number': idx + 1
                                })
                    
                    if time_comparison:
                        time_df = pd.DataFrame(time_comparison)
                        fig = px.line(
                            time_df, 
                            x='prediction_number', 
                            y='accuracy',
                            color='model',
                            title='Model Performance Over Time',
                            labels={'prediction_number': 'Prediction Number', 'accuracy': 'Rolling Accuracy'}
                        )
                        comparison_charts['performance_over_time'] = fig
            
            return {
                'comparison_metrics': make_arrow_compatible(pd.DataFrame(comparison_metrics)) if comparison_metrics else pd.DataFrame(),
                'model_rankings': model_rankings,
                'comparison_charts': comparison_charts,
                'summary': {
                    'total_models': len(model_groups),
                    'total_predictions': len(df),
                    'best_model': model_rankings[0]['model'] if model_rankings else 'N/A',
                    'best_model_score': model_rankings[0]['score'] if model_rankings else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating comparison dashboard: {str(e)}")
            return None

    def get_configuration(self) -> Dict[str, Any]:
        """Get current monitoring configuration."""
        return self.configuration.copy()
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update monitoring configuration."""
        try:
            # Validate and update configuration
            if 'drift_threshold' in new_config:
                threshold = float(new_config['drift_threshold'])
                if 0 < threshold <= 1:
                    self.configuration['drift_threshold'] = threshold
                    self.drift_threshold = threshold
            
            if 'monitoring_enabled' in new_config:
                self.configuration['monitoring_enabled'] = bool(new_config['monitoring_enabled'])
            
            if 'alert_threshold' in new_config:
                threshold = float(new_config['alert_threshold'])
                if 0 < threshold <= 1:
                    self.configuration['alert_threshold'] = threshold
            
            if 'min_samples_for_drift' in new_config:
                samples = int(new_config['min_samples_for_drift'])
                if samples > 0:
                    self.configuration['min_samples_for_drift'] = samples
            
            if 'performance_window' in new_config:
                window = int(new_config['performance_window'])
                if window > 0:
                    self.configuration['performance_window'] = window
                    
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")


def model_monitoring_dashboard():
    """Streamlit dashboard for model monitoring."""
    
    st.subheader("📈 Model Performance Monitoring")
    
    # Initialize monitor
    if 'model_monitor' not in st.session_state:
        st.session_state.model_monitor = ModelMonitor()
    
    monitor = st.session_state.model_monitor
    
    # Performance tracking section
    with st.expander("📊 Performance Tracking", expanded=True):
        
        if 'model_results' in st.session_state and st.session_state.model_results:
            
            # Show available models
            model_names = list(st.session_state.model_results.keys())
            selected_model = st.selectbox("Select Model to Monitor", model_names)
            
            if selected_model:
                model_info = st.session_state.model_results[selected_model]
                
                # Display model metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    accuracy = model_info.get('accuracy', model_info.get('r2_score', 0))
                    st.metric("Model Accuracy", f"{accuracy:.3f}")
                
                with col2:
                    training_time = model_info.get('training_time', 'N/A')
                    st.metric("Training Time", f"{training_time:.2f}s" if isinstance(training_time, (int, float)) else training_time)
                
                with col3:
                    feature_count = len(model_info.get('feature_importance', {}))
                    st.metric("Features Used", feature_count)
                
                with col4:
                    model_type = type(model_info.get('model', '')).__name__
                    st.metric("Model Type", model_type)
                
                # Performance trend simulation
                st.subheader("📈 Performance Trends")
                
                # Create sample performance data
                dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
                np.random.seed(42)
                performance_trend = 0.85 + 0.1 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.02, len(dates))
                
                trend_df = pd.DataFrame({
                    'Date': dates,
                    'Accuracy': performance_trend
                })
                
                fig = px.line(trend_df, x='Date', y='Accuracy', 
                             title=f"{selected_model} Performance Over Time")
                fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                             annotation_text="Alert Threshold")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Train a model first to enable performance monitoring")
    
    # Data drift detection section
    with st.expander("🔄 Data Drift Detection"):
        
        if 'cleaned_df' in st.session_state:
            df = st.session_state.cleaned_df
            
            st.write("**Simulate Data Drift Analysis**")
            
            # Create reference and current data splits
            if len(df) > 100:
                split_point = len(df) // 2
                reference_data = df.iloc[:split_point]
                current_data = df.iloc[split_point:]
                
                # Detect drift
                drift_results = monitor.detect_data_drift(current_data, reference_data)
                
                if drift_results:
                    st.subheader("🚨 Drift Detection Results")
                    
                    drift_df = pd.DataFrame([
                        {
                            'Column': col,
                            'Drift Score': results['drift_score'],
                            'Drift Detected': results['drift_detected'],
                            'Severity': results['severity']
                        }
                        for col, results in drift_results.items()
                    ])
                    
                    # Color code the dataframe
                    def color_drift(val):
                        if val == 'High':
                            return 'background-color: #ffebee'
                        elif val == 'Medium':
                            return 'background-color: #fff3e0'
                        return 'background-color: #e8f5e8'
                    
                    styled_df = drift_df.style.applymap(color_drift, subset=['Severity'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Drift visualization
                    high_drift_cols = [col for col, results in drift_results.items() 
                                     if results['severity'] == 'High']
                    
                    if high_drift_cols:
                        st.subheader("📊 High Drift Columns Analysis")
                        
                        for col in high_drift_cols[:3]:  # Show top 3
                            if pd.api.types.is_numeric_dtype(df[col]):
                                fig = go.Figure()
                                
                                fig.add_trace(go.Histogram(
                                    x=reference_data[col], 
                                    name='Reference Data',
                                    opacity=0.7,
                                    nbinsx=30
                                ))
                                
                                fig.add_trace(go.Histogram(
                                    x=current_data[col], 
                                    name='Current Data',
                                    opacity=0.7,
                                    nbinsx=30
                                ))
                                
                                fig.update_layout(
                                    title=f"Distribution Comparison: {col}",
                                    barmode='overlay'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Need more data points for meaningful drift analysis")
        
        else:
            st.info("👆 Load data first to enable drift detection")
    
    # Alerting section
    with st.expander("🚨 Alerting Configuration"):
        
        st.subheader("Alert Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            accuracy_threshold = st.slider(
                "Accuracy Alert Threshold", 
                min_value=0.5, 
                max_value=1.0, 
                value=0.8, 
                step=0.05,
                help="Alert when model accuracy drops below this value"
            )
        
        with col2:
            drift_threshold = st.slider(
                "Drift Alert Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.1, 
                step=0.05,
                help="Alert when data drift score exceeds this value"
            )
        
        monitor.drift_threshold = drift_threshold
        
        # Notification settings
        st.subheader("Notification Settings")
        
        notification_types = st.multiselect(
            "Alert Methods",
            ["Email", "Slack", "Dashboard", "Log File"],
            default=["Dashboard", "Log File"]
        )
        
        if st.button("💾 Save Alert Configuration"):
            config = {
                'accuracy_threshold': accuracy_threshold,
                'drift_threshold': drift_threshold,
                'notification_types': notification_types,
                'updated': datetime.now().isoformat()
            }
            
            # Save configuration
            with open('model_monitoring_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success("✅ Alert configuration saved!")
    
    # Model comparison section
    if 'model_results' in st.session_state and len(st.session_state.model_results) > 1:
        with st.expander("⚖️ Model Comparison"):
            
            st.subheader("Model Performance Comparison")
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, model_info in st.session_state.model_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': model_info.get('accuracy', model_info.get('r2_score', 0)),
                    'Training Time': model_info.get('training_time', 0),
                    'Features': len(model_info.get('feature_importance', {})),
                    'Type': type(model_info.get('model', '')).__name__
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Performance comparison chart
            fig = px.bar(comparison_df, x='Model', y='Accuracy', 
                        title="Model Accuracy Comparison",
                        color='Accuracy',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            st.dataframe(comparison_df, use_container_width=True)


def save_monitoring_report(monitor: ModelMonitor, filename: str = None):
    """Save monitoring report to file."""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_monitoring_report_{timestamp}.json"
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'performance_history': monitor.performance_history,
        'drift_threshold': monitor.drift_threshold,
        'summary': {
            'total_predictions': len(monitor.performance_history),
            'unique_models': len(set(entry['model_name'] for entry in monitor.performance_history))
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return filename
