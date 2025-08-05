# Enhanced modules/cleaning.py

import pandas as pd
import streamlit as st
import numpy as np
import logging
from scipy import stats

# Setup logger
logger = logging.getLogger(__name__)

def detect_outliers_iqr(series, multiplier=1.5):
    """Detect outliers using Interquartile Range (IQR) method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(series, threshold=3):
    """Detect outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(series.dropna()))
    # Handle the case where some values were dropped due to NaN
    outliers = pd.Series([False] * len(series), index=series.index)
    outliers.loc[series.dropna().index] = z_scores > threshold
    return outliers

def detect_outliers_modified_zscore(series, threshold=3.5):
    """Detect outliers using Modified Z-score method (more robust)."""
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        mad = np.median(np.abs(series - series.mean()))
    if mad == 0:
        return pd.Series([False] * len(series), index=series.index)
    
    modified_z_scores = 0.6745 * (series - median) / mad
    outliers = np.abs(modified_z_scores) > threshold
    return outliers

def handle_outliers(df: pd.DataFrame, method: str, detection_method: str, columns: list, **kwargs) -> pd.DataFrame:
    """
    Comprehensive outlier detection and treatment.
    
    Args:
        df: The input DataFrame
        method: Treatment method ('Remove', 'Cap', 'Transform', 'Replace with Mean', 'Replace with Median')
        detection_method: Detection method ('IQR', 'Z-Score', 'Modified Z-Score')
        columns: List of columns to process
        **kwargs: Additional parameters for detection methods
    
    Returns:
        DataFrame with outliers handled
    """
    try:
        df_cleaned = df.copy()
        outliers_info = {}
        
        if not columns:
            st.warning("Please select one or more columns for outlier detection.")
            return df_cleaned
        
        for col in columns:
            if col not in df.columns:
                st.warning(f"Column '{col}' not found in data.")
                continue
                
            if not pd.api.types.is_numeric_dtype(df[col]):
                st.warning(f"Column '{col}' is not numeric. Skipping outlier detection.")
                continue
            
            # Detect outliers based on selected method
            if detection_method == 'IQR':
                multiplier = kwargs.get('iqr_multiplier', 1.5)
                outliers, lower_bound, upper_bound = detect_outliers_iqr(df[col], multiplier)
                outliers_info[col] = {
                    'count': outliers.sum(),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'method': f'IQR (multiplier: {multiplier})'
                }
            elif detection_method == 'Z-Score':
                threshold = kwargs.get('zscore_threshold', 3)
                outliers = detect_outliers_zscore(df[col], threshold)
                outliers_info[col] = {
                    'count': outliers.sum(),
                    'method': f'Z-Score (threshold: {threshold})'
                }
            elif detection_method == 'Modified Z-Score':
                threshold = kwargs.get('modified_zscore_threshold', 3.5)
                outliers = detect_outliers_modified_zscore(df[col], threshold)
                outliers_info[col] = {
                    'count': outliers.sum(),
                    'method': f'Modified Z-Score (threshold: {threshold})'
                }
            
            # Apply treatment method
            if method == 'Remove':
                df_cleaned = df_cleaned[~outliers]
            elif method == 'Cap':
                if detection_method == 'IQR':
                    df_cleaned.loc[outliers & (df_cleaned[col] < lower_bound), col] = lower_bound
                    df_cleaned.loc[outliers & (df_cleaned[col] > upper_bound), col] = upper_bound
                else:
                    # For Z-score methods, cap at percentiles
                    lower_percentile = df[col].quantile(0.05)
                    upper_percentile = df[col].quantile(0.95)
                    df_cleaned.loc[outliers & (df_cleaned[col] < lower_percentile), col] = lower_percentile
                    df_cleaned.loc[outliers & (df_cleaned[col] > upper_percentile), col] = upper_percentile
            elif method == 'Transform':
                # Apply log transformation to reduce impact of outliers
                if (df_cleaned[col] > 0).all():
                    df_cleaned[col] = np.log1p(df_cleaned[col])
                else:
                    st.warning(f"Cannot apply log transformation to '{col}' due to non-positive values.")
            elif method == 'Replace with Mean':
                mean_val = df[col].mean()
                df_cleaned.loc[outliers, col] = mean_val
            elif method == 'Replace with Median':
                median_val = df[col].median()
                df_cleaned.loc[outliers, col] = median_val
        
        # Display outlier detection results
        if outliers_info:
            st.subheader("🎯 Outlier Detection Results")
            for col, info in outliers_info.items():
                st.write(f"**{col}**: {info['count']} outliers detected using {info['method']}")
                if 'lower_bound' in info and 'upper_bound' in info:
                    st.write(f"  - Lower bound: {info['lower_bound']:.2f}")
                    st.write(f"  - Upper bound: {info['upper_bound']:.2f}")
        
        logger.info(f"Outlier treatment completed using {detection_method} detection and {method} treatment")
        return df_cleaned
        
    except Exception as e:
        st.error(f"Error in outlier handling: {str(e)}")
        logger.error(f"Outlier handling error: {str(e)}")
        return df

def handle_missing_values(df: pd.DataFrame, strategy: str, columns: list) -> pd.DataFrame:
    """
    Enhanced missing value handling with better error handling and logging.

    Args:
        df: The input DataFrame.
        strategy: The imputation strategy ('Drop Rows', 'Fill with Mean', 
                  'Fill with Median', 'Fill with Mode', 'Forward Fill', 'Backward Fill').
        columns: The list of columns to apply the strategy to.

    Returns:
        A new DataFrame with missing values handled.
    """
    try:
        df_cleaned = df.copy()
        
        if not columns:
            st.warning("Please select one or more columns to apply the cleaning strategy.")
            return df_cleaned

        # Log the operation
        logger.info(f"Applying {strategy} to columns: {columns}")
        
        if strategy == "Drop Rows":
            initial_rows = len(df_cleaned)
            df_cleaned.dropna(subset=columns, inplace=True)
            rows_dropped = initial_rows - len(df_cleaned)
            logger.info(f"Dropped {rows_dropped} rows with missing values")
            
        elif strategy == "Forward Fill":
            for col in columns:
                df_cleaned[col].fillna(method='ffill', inplace=True)
                
        elif strategy == "Backward Fill":
            for col in columns:
                df_cleaned[col].fillna(method='bfill', inplace=True)
                
        else:
            for col in columns:
                if col not in df_cleaned.columns:
                    st.warning(f"Column '{col}' not found in dataframe.")
                    continue
                
                if strategy == "Fill with Mean":
                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        mean_value = df_cleaned[col].mean()
                        df_cleaned[col].fillna(mean_value, inplace=True)
                        logger.info(f"Filled {col} with mean: {mean_value:.2f}")
                    else:
                        st.warning(f"Cannot calculate mean for non-numeric column '{col}'. Skipping.")
                        
                elif strategy == "Fill with Median":
                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        median_value = df_cleaned[col].median()
                        df_cleaned[col].fillna(median_value, inplace=True)
                        logger.info(f"Filled {col} with median: {median_value:.2f}")
                    else:
                        st.warning(f"Cannot calculate median for non-numeric column '{col}'. Skipping.")
                        
                elif strategy == "Fill with Mode":
                    mode_values = df_cleaned[col].mode()
                    if not mode_values.empty:
                        mode_value = mode_values.iloc[0]
                        df_cleaned[col].fillna(mode_value, inplace=True)
                        logger.info(f"Filled {col} with mode: {mode_value}")
                    else:
                        st.warning(f"No mode found for column '{col}'. Skipping.")

        return df_cleaned
        
    except Exception as e:
        logger.error(f"Error in handle_missing_values: {str(e)}")
        st.error(f"Error handling missing values: {str(e)}")
        return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame with enhanced logging.

    Args:
        df: The input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    try:
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        rows_removed = initial_rows - len(df_cleaned)
        
        logger.info(f"Removed {rows_removed} duplicate rows")
        
        if rows_removed > 0:
            st.success(f"✅ Removed {rows_removed} duplicate rows")
        else:
            st.info("ℹ️ No duplicate rows found")
            
        return df_cleaned
        
    except Exception as e:
        logger.error(f"Error in remove_duplicates: {str(e)}")
        st.error(f"Error removing duplicates: {str(e)}")
        return df

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to CSV string for download.

    Args:
        df: The DataFrame to convert.

    Returns:
        CSV string.
    """
    try:
        return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        logger.error(f"Error converting DataFrame to CSV: {str(e)}")
        st.error(f"Error converting to CSV: {str(e)}")
        return ""
