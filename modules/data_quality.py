"""
Enhanced Data Quality and Cleaning Module
Comprehensive data validation and cleaning to prevent "Garbage In, Garbage Out"
"""

import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import re
from scipy import stats

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Comprehensive data quality assessment and validation."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []
        self.quality_score = 0
        self.recommendations = []
    
    def assess_overall_quality(self) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        
        # 1. Completeness Check
        completeness_score = self._check_completeness()
        
        # 2. Validity Check
        validity_score = self._check_validity()
        
        # 3. Consistency Check
        consistency_score = self._check_consistency()
        
        # 4. Uniqueness Check
        uniqueness_score = self._check_uniqueness()
        
        # 5. Accuracy Check (basic)
        accuracy_score = self._check_accuracy()
        
        # Calculate overall quality score
        weights = {
            'completeness': 0.25,
            'validity': 0.25,
            'consistency': 0.20,
            'uniqueness': 0.15,
            'accuracy': 0.15
        }
        
        self.quality_score = (
            completeness_score * weights['completeness'] +
            validity_score * weights['validity'] +
            consistency_score * weights['consistency'] +
            uniqueness_score * weights['uniqueness'] +
            accuracy_score * weights['accuracy']
        )
        
        return {
            'overall_score': self.quality_score,
            'completeness': completeness_score,
            'validity': validity_score,
            'consistency': consistency_score,
            'uniqueness': uniqueness_score,
            'accuracy': accuracy_score,
            'issues': self.issues,
            'recommendations': self.recommendations
        }
    
    def _check_completeness(self) -> float:
        """Check data completeness (missing values)."""
        total_cells = self.df.size
        missing_cells = self.df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        if missing_cells > 0:
            missing_by_col = self.df.isnull().sum()
            high_missing_cols = missing_by_col[missing_by_col > len(self.df) * 0.5]
            
            if len(high_missing_cols) > 0:
                self.issues.append(f"High missing values (>50%) in columns: {list(high_missing_cols.index)}")
                self.recommendations.append("Consider dropping columns with >70% missing values")
        
        return completeness
    
    def _check_validity(self) -> float:
        """Check data validity (data types, formats, ranges)."""
        validity_issues = 0
        total_checks = 0
        
        for col in self.df.columns:
            total_checks += 1
            
            # Check for mixed data types
            if self.df[col].dtype == 'object':
                # Check if numeric data is stored as string
                try:
                    pd.to_numeric(self.df[col].dropna())
                    self.issues.append(f"Column '{col}' contains numeric data stored as text")
                    self.recommendations.append(f"Convert '{col}' to numeric type")
                    validity_issues += 1
                except (ValueError, TypeError):
                    pass
                
                # Check for date-like strings
                if self._looks_like_date(col):
                    self.issues.append(f"Column '{col}' appears to contain dates stored as text")
                    self.recommendations.append(f"Convert '{col}' to datetime type")
                    validity_issues += 1
            
            # Check for negative values where they shouldn't be
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if col.lower() in ['age', 'salary', 'price', 'amount', 'quantity', 'count']:
                    negative_count = (self.df[col] < 0).sum()
                    if negative_count > 0:
                        self.issues.append(f"Column '{col}' has {negative_count} negative values")
                        validity_issues += 1
        
        validity_score = max(0, ((total_checks - validity_issues) / total_checks) * 100)
        return validity_score
    
    def _check_consistency(self) -> float:
        """Check data consistency (formats, naming conventions)."""
        consistency_issues = 0
        total_checks = len(self.df.columns)
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check for inconsistent text casing
                text_values = self.df[col].dropna().astype(str)
                if len(text_values) > 0:
                    mixed_case = any(val != val.lower() and val != val.upper() 
                                   for val in text_values if val.isalpha())
                    if mixed_case:
                        self.issues.append(f"Column '{col}' has inconsistent text casing")
                        self.recommendations.append(f"Standardize text casing in '{col}'")
                        consistency_issues += 1
                
                # Check for leading/trailing whitespace
                has_whitespace = text_values.str.strip().ne(text_values).any()
                if has_whitespace:
                    self.issues.append(f"Column '{col}' has leading/trailing whitespace")
                    self.recommendations.append(f"Trim whitespace in '{col}'")
                    consistency_issues += 1
        
        consistency_score = max(0, ((total_checks - consistency_issues) / total_checks) * 100)
        return consistency_score
    
    def _check_uniqueness(self) -> float:
        """Check for appropriate uniqueness in data."""
        uniqueness_issues = 0
        total_checks = 0
        
        # Check for potential ID columns that aren't unique
        for col in self.df.columns:
            if 'id' in col.lower() or col.lower().endswith('_key'):
                total_checks += 1
                if self.df[col].duplicated().any():
                    self.issues.append(f"ID column '{col}' contains duplicates")
                    self.recommendations.append(f"Investigate duplicate IDs in '{col}'")
                    uniqueness_issues += 1
        
        # Check for completely duplicate rows
        duplicate_rows = self.df.duplicated().sum()
        if duplicate_rows > 0:
            self.issues.append(f"Dataset contains {duplicate_rows} completely duplicate rows")
            self.recommendations.append("Remove duplicate rows")
        
        if total_checks == 0:
            return 100  # No ID columns to check
        
        uniqueness_score = max(0, ((total_checks - uniqueness_issues) / total_checks) * 100)
        return uniqueness_score
    
    def _check_accuracy(self) -> float:
        """Basic accuracy checks (outliers, impossible values)."""
        accuracy_issues = 0
        total_checks = 0
        
        for col in self.df.columns:
            # Skip boolean columns and only process true numeric columns
            if (pd.api.types.is_numeric_dtype(self.df[col]) and 
                not pd.api.types.is_bool_dtype(self.df[col])):
                total_checks += 1
                
                # Check for extreme outliers using IQR method
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers
                upper_bound = Q3 + 3 * IQR
                
                extreme_outliers = ((self.df[col] < lower_bound) | 
                                  (self.df[col] > upper_bound)).sum()
                
                if extreme_outliers > len(self.df) * 0.05:  # More than 5% extreme outliers
                    accuracy_issues += 1
                    self.issues.append(f"Column '{col}' has {extreme_outliers} extreme outliers")
                    self.recommendations.append(f"Investigate and handle outliers in '{col}'")
        
        if total_checks == 0:
            return 100  # No numeric columns to check
        
        accuracy_score = max(0, ((total_checks - accuracy_issues) / total_checks) * 100)
        return accuracy_score
    
    def _looks_like_date(self, col: str) -> bool:
        """Check if a text column looks like it contains dates."""
        sample_values = self.df[col].dropna().head(10)
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]
        
        for value in sample_values:
            if isinstance(value, str):
                for pattern in date_patterns:
                    if re.match(pattern, value):
                        return True
        return False


class SmartDataCleaner:
    """Intelligent data cleaning with contextual recommendations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cleaning_log = []
    
    def auto_clean_pipeline(self, aggressive: bool = False) -> pd.DataFrame:
        """Automated cleaning pipeline with smart defaults."""
        cleaned_df = self.df.copy()
        
        # 1. Remove completely empty rows and columns
        cleaned_df = self._remove_empty_rows_cols(cleaned_df)
        
        # 2. Standardize text data
        cleaned_df = self._standardize_text(cleaned_df)
        
        # 3. Fix data types
        cleaned_df = self._fix_data_types(cleaned_df)
        
        # 4. Handle duplicates intelligently
        cleaned_df = self._smart_duplicate_removal(cleaned_df, aggressive)
        
        # 5. Smart missing value imputation
        cleaned_df = self._smart_missing_value_handling(cleaned_df, aggressive)
        
        # 6. Handle outliers contextually
        if aggressive:
            cleaned_df = self._context_aware_outlier_handling(cleaned_df)
        
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
    
    def _standardize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text columns."""
        for col in df.select_dtypes(include=['object']).columns:
            # Remove leading/trailing whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Replace multiple spaces with single space
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            # Handle common text inconsistencies
            if col.lower() in ['gender', 'sex']:
                df[col] = df[col].str.lower().replace({
                    'm': 'male', 'f': 'female', 
                    'man': 'male', 'woman': 'female'
                })
            elif col.lower() in ['country', 'city', 'state']:
                df[col] = df[col].str.title()  # Proper case for places
        
        self.cleaning_log.append("Standardized text formatting")
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically fix obvious data type issues."""
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert numeric strings to numbers
                try:
                    # Remove common formatting (commas, dollar signs)
                    cleaned_values = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                    numeric_series = pd.to_numeric(cleaned_values, errors='coerce')
                    
                    # If most values convert successfully, use numeric type
                    if numeric_series.notna().sum() / len(df) > 0.8:
                        df[col] = numeric_series
                        self.cleaning_log.append(f"Converted '{col}' to numeric")
                
                except (ValueError, TypeError):
                    pass
                
                # Try to convert date strings
                if self._looks_like_date_column(df[col]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        self.cleaning_log.append(f"Converted '{col}' to datetime")
                    except:
                        pass
        
        return df
    
    def _smart_duplicate_removal(self, df: pd.DataFrame, aggressive: bool) -> pd.DataFrame:
        """Intelligent duplicate removal."""
        initial_count = len(df)
        
        if aggressive:
            # Remove any rows with identical values across all columns
            df = df.drop_duplicates()
        else:
            # Only remove completely identical rows (including index)
            df = df.drop_duplicates(keep='first')
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            self.cleaning_log.append(f"Removed {removed_count} duplicate rows")
        
        return df
    
    def _smart_missing_value_handling(self, df: pd.DataFrame, aggressive: bool) -> pd.DataFrame:
        """Context-aware missing value imputation."""
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            
            if missing_pct > 0.7 and aggressive:
                # Drop columns with >70% missing values
                df = df.drop(columns=[col])
                self.cleaning_log.append(f"Dropped column '{col}' (>70% missing)")
                continue
            
            if missing_pct > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Use median for skewed data, mean for normal data
                    if abs(df[col].skew()) > 1:
                        fill_value = df[col].median()
                        method = "median"
                    else:
                        fill_value = df[col].mean()
                        method = "mean"
                    df[col] = df[col].fillna(fill_value)
                else:
                    # Use mode for categorical data
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_value)
                    method = "mode"
                
                self.cleaning_log.append(f"Filled missing values in '{col}' using {method}")
        
        return df
    
    def _context_aware_outlier_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Context-aware outlier detection and handling."""
        for col in df.select_dtypes(include=[np.number]).columns:
            # Skip ID-like columns and boolean columns
            if 'id' in col.lower() or pd.api.types.is_bool_dtype(df[col]):
                continue
            
            # Use different methods based on column characteristics
            if col.lower() in ['age']:
                # Age has natural bounds
                df.loc[df[col] < 0, col] = np.nan
                df.loc[df[col] > 150, col] = np.nan
            elif col.lower() in ['salary', 'income', 'price']:
                # Remove negative values for monetary amounts
                df.loc[df[col] < 0, col] = np.nan
            
            # Apply statistical outlier detection
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outlier_count > 0:
                # Cap extreme outliers instead of removing
                df.loc[df[col] < lower_bound, col] = lower_bound
                df.loc[df[col] > upper_bound, col] = upper_bound
                self.cleaning_log.append(f"Capped {outlier_count} outliers in '{col}'")
        
        return df
    
    def _looks_like_date_column(self, series: pd.Series) -> bool:
        """Check if a series looks like it contains dates."""
        sample = series.dropna().head(5)
        date_count = 0
        
        for value in sample:
            try:
                pd.to_datetime(value)
                date_count += 1
            except:
                continue
        
        return date_count >= 3  # If 3+ samples look like dates


def enhanced_data_quality_dashboard(df: pd.DataFrame) -> None:
    """Enhanced data quality dashboard with actionable insights."""
    
    st.header("🔍 Advanced Data Quality Assessment")
    
    # Initialize quality checker
    quality_checker = DataQualityChecker(df)
    quality_report = quality_checker.assess_overall_quality()
    
    # Overall Quality Score
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        score = quality_report['overall_score']
        color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        st.metric(
            label=f"{color} Overall Data Quality Score", 
            value=f"{score:.1f}/100",
            help="Composite score based on completeness, validity, consistency, uniqueness, and accuracy"
        )
    
    with col2:
        if score >= 80:
            st.success("High Quality")
        elif score >= 60:
            st.warning("Medium Quality")
        else:
            st.error("Needs Attention")
    
    with col3:
        st.write(f"**Issues Found:** {len(quality_report['issues'])}")
    
    # Detailed Quality Metrics
    st.subheader("📊 Quality Dimensions")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4, metrics_col5 = st.columns(5)
    
    with metrics_col1:
        st.metric("Completeness", f"{quality_report['completeness']:.1f}%")
    with metrics_col2:
        st.metric("Validity", f"{quality_report['validity']:.1f}%")
    with metrics_col3:
        st.metric("Consistency", f"{quality_report['consistency']:.1f}%")
    with metrics_col4:
        st.metric("Uniqueness", f"{quality_report['uniqueness']:.1f}%")
    with metrics_col5:
        st.metric("Accuracy", f"{quality_report['accuracy']:.1f}%")
    
    # Issues and Recommendations
    if quality_report['issues']:
        st.subheader("⚠️ Data Quality Issues")
        for i, issue in enumerate(quality_report['issues'], 1):
            st.write(f"{i}. {issue}")
    
    if quality_report['recommendations']:
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(quality_report['recommendations'], 1):
            st.write(f"{i}. {rec}")
    
    # Smart Cleaning Options
    # Note: Auto-cleaning buttons moved to main app.py Enhanced Auto-Clean Pipeline section
    # This keeps the data quality dashboard focused on assessment and reporting
