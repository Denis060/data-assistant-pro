"""
Time Series Analysis Module
Advanced temporal data analysis, forecasting, and visualization
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
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("Statsmodels not available. Some time series features will be limited.")


class TimeSeriesAnalyzer:
    """Advanced time series analysis and forecasting."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.date_columns = self._detect_date_columns()
        self.numeric_columns = self._detect_numeric_columns()
        
    def _detect_date_columns(self) -> List[str]:
        """Detect potential date columns."""
        date_cols = []
        
        for col in self.data.columns:
            # Check if already datetime
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                date_cols.append(col)
                continue
                
            # Check if column name suggests date
            date_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'year', 'month']
            if any(keyword in col.lower() for keyword in date_keywords):
                try:
                    pd.to_datetime(self.data[col].dropna().head(100))
                    date_cols.append(col)
                except:
                    pass
                    
        return date_cols
    
    def _detect_numeric_columns(self) -> List[str]:
        """Detect numeric columns suitable for time series analysis."""
        return list(self.data.select_dtypes(include=[np.number]).columns)
    
    def prepare_time_series(self, date_col: str, value_col: str, freq: str = 'D') -> pd.Series:
        """Prepare time series data."""
        try:
            # Convert date column to datetime
            if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
                dates = pd.to_datetime(self.data[date_col])
            else:
                dates = self.data[date_col]
            
            # Create time series
            ts_data = pd.Series(
                data=self.data[value_col].values,
                index=dates,
                name=value_col
            )
            
            # Sort by date
            ts_data = ts_data.sort_index()
            
            # Remove duplicates by averaging
            ts_data = ts_data.groupby(ts_data.index).mean()
            
            # Resample to specified frequency
            if freq != 'Original':
                ts_data = ts_data.resample(freq).mean()
            
            return ts_data.dropna()
            
        except Exception as e:
            logger.error(f"Error preparing time series: {str(e)}")
            raise e
    
    def analyze_stationarity(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Analyze stationarity using Augmented Dickey-Fuller test."""
        if not STATSMODELS_AVAILABLE:
            return {"error": "Statsmodels not available"}
        
        try:
            result = adfuller(ts_data.dropna())
            
            return {
                'adf_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05,
                'interpretation': 'Stationary' if result[1] < 0.05 else 'Non-stationary'
            }
        except Exception as e:
            logger.error(f"Error in stationarity test: {str(e)}")
            return {"error": str(e)}
    
    def decompose_time_series(self, ts_data: pd.Series, model: str = 'additive', period: int = None) -> Dict[str, pd.Series]:
        """Decompose time series into trend, seasonal, and residual components."""
        if not STATSMODELS_AVAILABLE:
            return {"error": "Statsmodels not available"}
        
        try:
            if period is None:
                # Try to detect seasonality automatically
                if len(ts_data) >= 24:
                    period = min(12, len(ts_data) // 2)
                else:
                    period = len(ts_data) // 2 if len(ts_data) > 4 else 2
            
            decomposition = seasonal_decompose(ts_data, model=model, period=period)
            
            return {
                'trend': decomposition.trend.dropna(),
                'seasonal': decomposition.seasonal.dropna(),
                'residual': decomposition.resid.dropna(),
                'original': ts_data
            }
        except Exception as e:
            logger.error(f"Error in decomposition: {str(e)}")
            return {"error": str(e)}
    
    def forecast_arima(self, ts_data: pd.Series, order: Tuple[int, int, int] = (1, 1, 1), steps: int = 30) -> Dict[str, Any]:
        """ARIMA forecasting."""
        if not STATSMODELS_AVAILABLE:
            return {"error": "Statsmodels not available"}
        
        try:
            model = ARIMA(ts_data, order=order)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=steps)
            conf_int = fitted_model.get_forecast(steps=steps).conf_int()
            
            # Create forecast dates
            last_date = ts_data.index[-1]
            freq = pd.infer_freq(ts_data.index)
            if freq is None:
                freq = 'D'  # Default to daily
            
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq=freq
            )
            
            return {
                'forecast': pd.Series(forecast, index=forecast_dates),
                'lower_bound': pd.Series(conf_int.iloc[:, 0], index=forecast_dates),
                'upper_bound': pd.Series(conf_int.iloc[:, 1], index=forecast_dates),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'model_summary': str(fitted_model.summary())
            }
        except Exception as e:
            logger.error(f"Error in ARIMA forecasting: {str(e)}")
            return {"error": str(e)}
    
    def forecast_exponential_smoothing(self, ts_data: pd.Series, steps: int = 30) -> Dict[str, Any]:
        """Exponential Smoothing forecasting."""
        if not STATSMODELS_AVAILABLE:
            return {"error": "Statsmodels not available"}
        
        try:
            model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=12)
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=steps)
            
            # Create forecast dates
            last_date = ts_data.index[-1]
            freq = pd.infer_freq(ts_data.index)
            if freq is None:
                freq = 'D'
            
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq=freq
            )
            
            return {
                'forecast': pd.Series(forecast, index=forecast_dates),
                'aic': fitted_model.aic,
                'model_summary': 'Exponential Smoothing model fitted successfully'
            }
        except Exception as e:
            logger.error(f"Error in Exponential Smoothing: {str(e)}")
            return {"error": str(e)}
    
    def calculate_moving_averages(self, ts_data: pd.Series, windows: List[int] = [7, 30, 90]) -> Dict[str, pd.Series]:
        """Calculate moving averages."""
        moving_averages = {}
        
        for window in windows:
            if len(ts_data) >= window:
                ma = ts_data.rolling(window=window).mean()
                moving_averages[f'MA_{window}'] = ma
        
        return moving_averages
    
    def detect_anomalies(self, ts_data: pd.Series, method: str = 'iqr', threshold: float = 2.0) -> pd.Series:
        """Detect anomalies in time series data."""
        if method == 'iqr':
            Q1 = ts_data.quantile(0.25)
            Q3 = ts_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies = (ts_data < lower_bound) | (ts_data > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((ts_data - ts_data.mean()) / ts_data.std())
            anomalies = z_scores > threshold
            
        else:  # Standard deviation
            mean = ts_data.mean()
            std = ts_data.std()
            anomalies = np.abs(ts_data - mean) > threshold * std
        
        return anomalies


def time_series_dashboard():
    """Streamlit dashboard for time series analysis."""
    
    st.subheader("📅 Time Series Analysis")
    
    if 'cleaned_df' not in st.session_state:
        st.warning("⚠️ Please load data first in the main application.")
        return
    
    df = st.session_state.cleaned_df
    analyzer = TimeSeriesAnalyzer(df)
    
    # Check if we have date columns
    if not analyzer.date_columns:
        st.error("❌ No date columns detected in your data. Time series analysis requires at least one date column.")
        st.info("💡 Tip: Make sure your date columns have names like 'date', 'timestamp', 'created_at', etc.")
        return
    
    if not analyzer.numeric_columns:
        st.error("❌ No numeric columns found for analysis.")
        return
    
    # Configuration section
    with st.expander("⚙️ Time Series Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date_col = st.selectbox("📅 Date Column", analyzer.date_columns)
        
        with col2:
            value_col = st.selectbox("📊 Value Column", analyzer.numeric_columns)
        
        with col3:
            freq = st.selectbox(
                "🔄 Frequency",
                ['Original', 'D', 'W', 'M', 'Q', 'Y'],
                help="D=Daily, W=Weekly, M=Monthly, Q=Quarterly, Y=Yearly"
            )
    
    if not date_col or not value_col:
        st.warning("⚠️ Please select both date and value columns.")
        return
    
    try:
        # Prepare time series data
        with st.spinner("📈 Preparing time series data..."):
            ts_data = analyzer.prepare_time_series(date_col, value_col, freq)
        
        if len(ts_data) < 2:
            st.error("❌ Not enough data points for time series analysis.")
            return
        
        st.success(f"✅ Time series prepared: {len(ts_data)} data points")
        
        # Analysis tabs
        ts_tab1, ts_tab2, ts_tab3, ts_tab4 = st.tabs([
            "📊 Visualization", "🔄 Decomposition", "🔮 Forecasting", "⚠️ Anomaly Detection"
        ])
        
        with ts_tab1:
            st.subheader("📊 Time Series Visualization")
            
            # Basic time series plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts_data.index, 
                y=ts_data.values,
                mode='lines',
                name=value_col,
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add moving averages
            moving_averages = analyzer.calculate_moving_averages(ts_data)
            colors = ['#ff7f0e', '#2ca02c', '#d62728']
            
            for i, (name, ma_data) in enumerate(moving_averages.items()):
                fig.add_trace(go.Scatter(
                    x=ma_data.index,
                    y=ma_data.values,
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=1, dash='dash')
                ))
            
            fig.update_layout(
                title=f"Time Series: {value_col}",
                xaxis_title="Date",
                yaxis_title=value_col,
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Points", f"{len(ts_data):,}")
            with col2:
                st.metric("Mean", f"{ts_data.mean():.2f}")
            with col3:
                st.metric("Std Dev", f"{ts_data.std():.2f}")
            with col4:
                st.metric("Range", f"{ts_data.max() - ts_data.min():.2f}")
            
            # Stationarity test
            if st.button("🔍 Test Stationarity"):
                with st.spinner("Running Augmented Dickey-Fuller test..."):
                    stationarity_result = analyzer.analyze_stationarity(ts_data)
                    
                if 'error' not in stationarity_result:
                    col1, col2 = st.columns(2)
                    with col1:
                        if stationarity_result['is_stationary']:
                            st.success("✅ Time series is stationary")
                        else:
                            st.warning("⚠️ Time series is non-stationary")
                    
                    with col2:
                        st.metric("P-value", f"{stationarity_result['p_value']:.4f}")
                else:
                    st.error(f"❌ Stationarity test failed: {stationarity_result['error']}")
        
        with ts_tab2:
            st.subheader("🔄 Time Series Decomposition")
            
            if not STATSMODELS_AVAILABLE:
                st.error("❌ Statsmodels not available. Please install: pip install statsmodels")
                return
            
            col1, col2 = st.columns(2)
            with col1:
                decomp_model = st.selectbox("Model Type", ['additive', 'multiplicative'])
            with col2:
                period = st.number_input("Seasonal Period", min_value=2, value=12, help="Number of periods in a season")
            
            if st.button("🔄 Decompose Time Series"):
                with st.spinner("Decomposing time series..."):
                    decomposition = analyzer.decompose_time_series(ts_data, decomp_model, period)
                
                if 'error' not in decomposition:
                    # Create subplots for decomposition
                    fig = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                        vertical_spacing=0.05
                    )
                    
                    components = ['original', 'trend', 'seasonal', 'residual']
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    
                    for i, (comp, color) in enumerate(zip(components, colors)):
                        if comp in decomposition:
                            data = decomposition[comp]
                            fig.add_trace(
                                go.Scatter(x=data.index, y=data.values, name=comp.title(), line=dict(color=color)),
                                row=i+1, col=1
                            )
                    
                    fig.update_layout(height=800, title="Time Series Decomposition", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"❌ Decomposition failed: {decomposition['error']}")
        
        with ts_tab3:
            st.subheader("🔮 Time Series Forecasting")
            
            if not STATSMODELS_AVAILABLE:
                st.error("❌ Statsmodels not available. Please install: pip install statsmodels")
                return
            
            col1, col2 = st.columns(2)
            with col1:
                forecast_method = st.selectbox("Forecasting Method", ['ARIMA', 'Exponential Smoothing'])
            with col2:
                forecast_steps = st.number_input("Forecast Steps", min_value=1, max_value=365, value=30)
            
            if forecast_method == 'ARIMA':
                st.write("**ARIMA Parameters:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
                with col2:
                    d = st.number_input("Differencing (d)", min_value=0, max_value=2, value=1)
                with col3:
                    q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1)
            
            if st.button("🔮 Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    if forecast_method == 'ARIMA':
                        forecast_result = analyzer.forecast_arima(ts_data, order=(p, d, q), steps=forecast_steps)
                    else:
                        forecast_result = analyzer.forecast_exponential_smoothing(ts_data, steps=forecast_steps)
                
                if 'error' not in forecast_result:
                    # Plot forecast
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=ts_data.index, 
                        y=ts_data.values,
                        mode='lines',
                        name='Historical',
                        line=dict(color='#1f77b4')
                    ))
                    
                    # Forecast
                    forecast_data = forecast_result['forecast']
                    fig.add_trace(go.Scatter(
                        x=forecast_data.index,
                        y=forecast_data.values,
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#ff7f0e', dash='dash')
                    ))
                    
                    # Confidence intervals for ARIMA
                    if 'lower_bound' in forecast_result:
                        fig.add_trace(go.Scatter(
                            x=forecast_result['lower_bound'].index,
                            y=forecast_result['lower_bound'].values,
                            mode='lines',
                            name='Lower Bound',
                            line=dict(color='rgba(255,127,14,0.3)'),
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_result['upper_bound'].index,
                            y=forecast_result['upper_bound'].values,
                            mode='lines',
                            name='Upper Bound',
                            line=dict(color='rgba(255,127,14,0.3)'),
                            fill='tonexty',
                            showlegend=False
                        ))
                    
                    fig.update_layout(
                        title=f"{forecast_method} Forecast",
                        xaxis_title="Date",
                        yaxis_title=value_col,
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        if 'aic' in forecast_result:
                            st.metric("AIC", f"{forecast_result['aic']:.2f}")
                    with col2:
                        if 'bic' in forecast_result:
                            st.metric("BIC", f"{forecast_result['bic']:.2f}")
                    
                    # Download forecast
                    csv = forecast_data.to_csv()
                    st.download_button(
                        "📥 Download Forecast",
                        csv,
                        file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
                else:
                    st.error(f"❌ Forecasting failed: {forecast_result['error']}")
        
        with ts_tab4:
            st.subheader("⚠️ Anomaly Detection")
            
            col1, col2 = st.columns(2)
            with col1:
                anomaly_method = st.selectbox("Detection Method", ['iqr', 'zscore', 'std'])
            with col2:
                if anomaly_method in ['zscore', 'std']:
                    threshold = st.number_input("Threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
                else:
                    threshold = 1.5  # Fixed for IQR
            
            if st.button("🔍 Detect Anomalies"):
                with st.spinner("Detecting anomalies..."):
                    anomalies = analyzer.detect_anomalies(ts_data, anomaly_method, threshold)
                
                anomaly_count = anomalies.sum()
                st.metric("Anomalies Detected", f"{anomaly_count} ({anomaly_count/len(ts_data)*100:.1f}%)")
                
                if anomaly_count > 0:
                    # Plot anomalies
                    fig = go.Figure()
                    
                    # Normal data
                    normal_data = ts_data[anomalies == False]
                    fig.add_trace(go.Scatter(
                        x=normal_data.index,
                        y=normal_data.values,
                        mode='lines+markers',
                        name='Normal',
                        line=dict(color='#1f77b4'),
                        marker=dict(size=4)
                    ))
                    
                    # Anomalies
                    anomaly_data = ts_data[anomalies]
                    fig.add_trace(go.Scatter(
                        x=anomaly_data.index,
                        y=anomaly_data.values,
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=8, symbol='x')
                    ))
                    
                    fig.update_layout(
                        title="Anomaly Detection Results",
                        xaxis_title="Date",
                        yaxis_title=value_col,
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show anomaly details
                    if anomaly_count <= 20:  # Show details for reasonable number
                        st.subheader("🔍 Anomaly Details")
                        anomaly_details = pd.DataFrame({
                            'Date': anomaly_data.index,
                            'Value': anomaly_data.values
                        })
                        st.dataframe(anomaly_details, use_container_width=True)
                else:
                    st.success("✅ No anomalies detected!")
    
    except Exception as e:
        st.error(f"❌ Error in time series analysis: {str(e)}")
        logger.error(f"Time series analysis error: {str(e)}")


def time_series_features_info():
    """Display information about time series features."""
    
    st.markdown("""
    ### 📅 Time Series Analysis Features
    
    **🔍 Data Preparation:**
    - Automatic date column detection
    - Data frequency conversion (Daily, Weekly, Monthly, etc.)
    - Missing value handling
    - Duplicate timestamp resolution
    
    **📊 Visualization:**
    - Interactive time series plots
    - Moving averages (7, 30, 90 days)
    - Trend analysis
    - Statistical summaries
    
    **🔄 Decomposition:**
    - Seasonal decomposition (additive/multiplicative)
    - Trend extraction
    - Seasonal pattern analysis
    - Residual analysis
    
    **🔮 Forecasting:**
    - ARIMA models with customizable parameters
    - Exponential Smoothing
    - Confidence intervals
    - Model performance metrics (AIC, BIC)
    
    **⚠️ Anomaly Detection:**
    - Multiple detection methods (IQR, Z-score, Standard Deviation)
    - Configurable thresholds
    - Visual anomaly highlighting
    - Detailed anomaly reports
    
    **🔍 Stationarity Testing:**
    - Augmented Dickey-Fuller test
    - Statistical significance assessment
    - Stationarity recommendations
    """)
