# Enhanced modules/cleaning.py

import pandas as pd
import streamlit as st
import numpy as np
import logging

# Setup logger
logger = logging.getLogger(__name__)

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
