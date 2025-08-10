"""
Enhanced SmartDataCleaner with improved algorithms
Building on your existing excellent foundation
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class EnhancedSmartDataCleaner:
    """
    Improved version of your SmartDataCleaner with enhanced algorithms
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cleaning_log = []
    
    def auto_clean_pipeline(self, aggressive: bool = False) -> pd.DataFrame:
        """Enhanced automated cleaning pipeline with smarter algorithms."""
        cleaned_df = self.df.copy()
        
        # 1. Remove completely empty rows and columns
        cleaned_df = self._remove_empty_rows_cols(cleaned_df)
        
        # 2. Enhanced text standardization
        cleaned_df = self._enhanced_text_standardization(cleaned_df)
        
        # 3. Smarter data type fixes
        cleaned_df = self._smart_data_type_conversion(cleaned_df)
        
        # 4. Enhanced duplicate removal
        cleaned_df = self._enhanced_duplicate_removal(cleaned_df, aggressive)
        
        # 5. Context-aware missing value handling
        cleaned_df = self._context_aware_missing_value_handling(cleaned_df, aggressive)
        
        # 6. Improved outlier handling
        if aggressive:
            cleaned_df = self._enhanced_outlier_handling(cleaned_df)
        
        # 7. Categorical standardization
        cleaned_df = self._standardize_categorical_values(cleaned_df)
        
        return cleaned_df
    
    def _remove_empty_rows_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove completely empty rows and columns."""
        initial_shape = df.shape
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Remove completely empty rows
        df = df.dropna(axis=0, how='all')
        
        if df.shape != initial_shape:
            self.cleaning_log.append(
                f"Removed empty rows/columns: {initial_shape} → {df.shape}"
            )
        
        return df
    
    def _enhanced_text_standardization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced text standardization with more comprehensive rules."""
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                # Convert to string and handle NaN properly
                df[col] = df[col].astype(str)
                
                # Remove leading/trailing whitespace
                df[col] = df[col].str.strip()
                
                # Replace multiple spaces with single space
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                
                # Handle 'nan' strings created by astype(str)
                df[col] = df[col].replace('nan', np.nan)
                
                # Enhanced standardization rules
                if col.lower() in ['gender', 'sex']:
                    df[col] = df[col].str.lower().replace({
                        'm': 'male', 'f': 'female', 'man': 'male', 'woman': 'female',
                        'male': 'male', 'female': 'female'
                    })
                elif 'phone' in col.lower() or 'service' in col.lower():
                    df[col] = df[col].str.lower().replace({
                        'y': 'yes', 'n': 'no', '1': 'yes', '0': 'no',
                        'true': 'yes', 'false': 'no'
                    })
                elif 'contract' in col.lower() or 'type' in col.lower():
                    # Standardize contract types
                    df[col] = df[col].str.replace('monthly', 'Month-to-month', case=False)
                    df[col] = df[col].str.replace('1-year', 'One year', case=False)
                    df[col] = df[col].str.replace('2-year', 'Two year', case=False)
                elif col.lower() in ['country', 'city', 'state', 'location']:
                    df[col] = df[col].str.title()  # Proper case for places
        
        self.cleaning_log.append("Enhanced text standardization completed")
        return df
    
    def _smart_data_type_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smarter data type conversion with better error handling."""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert numeric strings to numbers (handle currency, commas)
                try:
                    # Remove common formatting characters
                    cleaned_values = df[col].astype(str).str.replace(r'[$,€£¥]', '', regex=True)
                    cleaned_values = cleaned_values.str.replace(r'[^\d.-]', '', regex=True)
                    
                    # Try numeric conversion
                    numeric_series = pd.to_numeric(cleaned_values, errors='coerce')
                    
                    # Only convert if >80% of non-null values are numeric
                    non_null_count = df[col].notna().sum()
                    numeric_count = numeric_series.notna().sum()
                    
                    if non_null_count > 0 and (numeric_count / non_null_count) > 0.8:
                        df[col] = numeric_series
                        self.cleaning_log.append(f"Converted '{col}' to numeric (currency formatting removed)")
                
                except (ValueError, TypeError):
                    pass
                
                # Try to convert date strings with multiple formats
                if self._looks_like_date_column(df[col]):
                    try:
                        # Try multiple date formats
                        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']
                        converted = False
                        
                        for fmt in date_formats:
                            try:
                                df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
                                if df[col].notna().sum() > len(df) * 0.8:
                                    self.cleaning_log.append(f"Converted '{col}' to datetime")
                                    converted = True
                                    break
                            except:
                                continue
                        
                        if not converted:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass
        
        return df
    
    def _enhanced_duplicate_removal(self, df: pd.DataFrame, aggressive: bool) -> pd.DataFrame:
        """Enhanced duplicate removal with intelligent detection."""
        initial_count = len(df)
        
        if aggressive:
            # More sophisticated duplicate detection
            # Consider rows duplicate if all non-ID columns are identical
            non_id_cols = [col for col in df.columns if 'id' not in col.lower()]
            df = df.drop_duplicates(subset=non_id_cols, keep='first')
        else:
            # Conservative: only remove exact duplicates
            df = df.drop_duplicates(keep='first')
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            mode = "aggressive" if aggressive else "conservative"
            self.cleaning_log.append(f"Removed {removed_count} duplicate rows ({mode} mode)")
        
        return df
    
    def _context_aware_missing_value_handling(self, df: pd.DataFrame, aggressive: bool) -> pd.DataFrame:
        """Context-aware missing value imputation with domain knowledge."""
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            
            # Drop columns with excessive missing values (aggressive mode only)
            if missing_pct > 0.7 and aggressive:
                df = df.drop(columns=[col])
                self.cleaning_log.append(f"Dropped column '{col}' ({missing_pct:.1%} missing)")
                continue
            
            if missing_pct > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Context-aware numeric imputation
                    if col.lower() in ['age']:
                        # Age: use median, as it's less affected by outliers
                        fill_value = df[col].median()
                        method = "median (age-specific)"
                    elif col.lower() in ['salary', 'income', 'price', 'cost', 'charges']:
                        # Financial: use median for skewed distributions
                        fill_value = df[col].median()
                        method = "median (financial)"
                    elif col.lower() in ['count', 'quantity', 'number']:
                        # Counts: use mode or 0
                        fill_value = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                        method = "mode (count)"
                    else:
                        # General numeric: choose based on skewness
                        skewness = abs(df[col].skew())
                        if skewness > 1:
                            fill_value = df[col].median()
                            method = "median (skewed)"
                        else:
                            fill_value = df[col].mean()
                            method = "mean (normal)"
                    
                    df[col] = df[col].fillna(fill_value)
                    
                else:
                    # Categorical imputation
                    if missing_pct < 0.1:  # Low missing rate
                        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                        method = "mode"
                    else:  # High missing rate
                        mode_value = 'Unknown'
                        method = "unknown placeholder"
                    
                    df[col] = df[col].fillna(mode_value)
                
                self.cleaning_log.append(f"Filled missing values in '{col}' using {method}")
        
        return df
    
    def _enhanced_outlier_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced context-aware outlier detection and handling."""
        for col in df.select_dtypes(include=[np.number]).columns:
            # Skip ID-like columns and boolean columns
            if 'id' in col.lower() or pd.api.types.is_bool_dtype(df[col]):
                continue
            
            original_outliers = 0
            
            # Domain-specific outlier handling
            if col.lower() in ['age']:
                # Age bounds: 0-120 years
                outliers = (df[col] < 0) | (df[col] > 120)
                original_outliers = outliers.sum()
                df.loc[df[col] < 0, col] = 0
                df.loc[df[col] > 120, col] = 120
                
            elif col.lower() in ['salary', 'income', 'price', 'charges', 'cost']:
                # Remove negative financial values
                outliers = df[col] < 0
                original_outliers = outliers.sum()
                df.loc[df[col] < 0, col] = np.nan  # Will be filled by missing value handler
                
            elif col.lower() in ['tenure', 'months', 'years']:
                # Tenure bounds
                outliers = (df[col] < 0) | (df[col] > 100)
                original_outliers = outliers.sum()
                df.loc[df[col] < 0, col] = 0
                df.loc[df[col] > 100, col] = 100
            
            # Apply statistical outlier detection (IQR method)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Avoid division by zero
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                statistical_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                statistical_count = statistical_outliers.sum()
                
                # Cap extreme outliers instead of removing
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                
                total_outliers = original_outliers + statistical_count
                if total_outliers > 0:
                    self.cleaning_log.append(f"Handled {total_outliers} outliers in '{col}' (domain + statistical)")
        
        return df
    
    def _standardize_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical values for consistency."""
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                # Get value counts
                value_counts = df[col].value_counts()
                
                # If we have many similar values, try to consolidate
                unique_values = df[col].dropna().unique()
                
                if len(unique_values) > 2:
                    # Create mapping for similar values
                    standardization_map = {}
                    
                    for value in unique_values:
                        if isinstance(value, str):
                            lower_val = value.lower().strip()
                            
                            # Binary mappings
                            if lower_val in ['yes', 'y', '1', 'true', 'on']:
                                standardization_map[value] = 'Yes'
                            elif lower_val in ['no', 'n', '0', 'false', 'off']:
                                standardization_map[value] = 'No'
                    
                    if standardization_map:
                        df[col] = df[col].replace(standardization_map)
                        self.cleaning_log.append(f"Standardized categorical values in '{col}'")
        
        return df
    
    def _looks_like_date_column(self, series: pd.Series) -> bool:
        """Enhanced date detection."""
        sample = series.dropna().head(10)
        date_count = 0
        
        for value in sample:
            if isinstance(value, str):
                # Common date patterns
                import re
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                    r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                    r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                    r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
                ]
                
                for pattern in date_patterns:
                    if re.search(pattern, value):
                        date_count += 1
                        break
        
        return date_count >= 3  # If 3+ samples look like dates
