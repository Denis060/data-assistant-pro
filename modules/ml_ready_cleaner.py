"""
ML-Ready Data Cleaner
Advanced cleaning specifically designed for machine learning success
Ensures data quality that leads to robust, high-performing models
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
import logging
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class MLReadyCleaner:
    """
    Advanced data cleaner specifically designed for machine learning pipelines.
    Focuses on creating consistent, high-quality data that leads to better model performance.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = None):
        self.df = df.copy()
        self.target_column = target_column
        self.cleaning_log = []
        self.data_quality_issues = []
        self.ml_recommendations = []
        
    def prepare_for_ml(self, problem_type: str = 'auto', test_size: float = 0.2) -> dict:
        """
        Complete ML-ready data preparation pipeline.
        
        Args:
            problem_type: 'classification', 'regression', or 'auto'
            test_size: Proportion for train/test split validation
            
        Returns:
            dict: Cleaned data, encoders, scalers, and quality report
        """
        
        # 1. Initial data assessment
        self._assess_data_quality()
        
        # 2. Handle critical data quality issues
        cleaned_df = self._handle_critical_issues()
        
        # 3. Intelligent missing value handling
        cleaned_df = self._ml_missing_value_strategy(cleaned_df)
        
        # 4. Advanced outlier detection and treatment
        cleaned_df = self._ml_outlier_treatment(cleaned_df)
        
        # 5. Feature engineering for ML
        cleaned_df = self._ml_feature_engineering(cleaned_df)
        
        # 6. Data type optimization for ML
        cleaned_df, encoders = self._optimize_for_ml(cleaned_df)
        
        # 7. Feature validation and selection
        cleaned_df = self._validate_features_for_ml(cleaned_df)
        
        # 8. Final ML readiness check
        ml_report = self._generate_ml_readiness_report(cleaned_df)
        
        return {
            'cleaned_data': cleaned_df,
            'encoders': encoders,
            'cleaning_log': self.cleaning_log,
            'quality_issues': self.data_quality_issues,
            'ml_recommendations': self.ml_recommendations,
            'ml_readiness_report': ml_report
        }
    
    def _assess_data_quality(self):
        """Comprehensive data quality assessment for ML."""
        df = self.df
        
        # Check data shape and basic info
        self.cleaning_log.append(f"📊 Initial data shape: {df.shape}")
        
        # Missing data analysis
        missing_pct = (df.isnull().sum() / len(df)) * 100
        critical_missing = missing_pct[missing_pct > 50]
        
        if len(critical_missing) > 0:
            self.data_quality_issues.append({
                'type': 'Critical Missing Data',
                'severity': 'High',
                'columns': critical_missing.index.tolist(),
                'description': f"{len(critical_missing)} columns have >50% missing values"
            })
        
        # Duplicate analysis
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(df)) * 100
            severity = 'High' if duplicate_pct > 10 else 'Medium'
            self.data_quality_issues.append({
                'type': 'Duplicate Records',
                'severity': severity,
                'count': duplicate_count,
                'percentage': duplicate_pct,
                'description': f"{duplicate_count} duplicate rows ({duplicate_pct:.1f}%)"
            })
        
        # Data type inconsistencies
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data is stored as text
                try:
                    numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                    numeric_pct = numeric_conversion.notna().sum() / df[col].notna().sum()
                    
                    if numeric_pct > 0.8:  # 80% can be converted to numeric
                        self.data_quality_issues.append({
                            'type': 'Data Type Mismatch',
                            'severity': 'Medium',
                            'column': col,
                            'description': f"Column '{col}' appears numeric but stored as text"
                        })
                except:
                    pass
        
        # Target variable analysis
        if self.target_column and self.target_column in df.columns:
            target_missing = df[self.target_column].isnull().sum()
            if target_missing > 0:
                self.data_quality_issues.append({
                    'type': 'Target Variable Issues',
                    'severity': 'Critical',
                    'description': f"Target variable '{self.target_column}' has {target_missing} missing values"
                })
        
        self.cleaning_log.append(f"🔍 Identified {len(self.data_quality_issues)} data quality issues")
    
    def _handle_critical_issues(self) -> pd.DataFrame:
        """Handle critical data quality issues that would break ML training."""
        df = self.df.copy()
        
        # Remove completely empty rows and columns
        initial_shape = df.shape
        df = df.dropna(how='all')  # Remove empty rows
        df = df.dropna(axis=1, how='all')  # Remove empty columns
        
        if df.shape != initial_shape:
            self.cleaning_log.append(f"🧹 Removed empty rows/columns: {initial_shape} → {df.shape}")
        
        # Handle target variable issues
        if self.target_column and self.target_column in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=[self.target_column])
            removed = initial_len - len(df)
            if removed > 0:
                self.cleaning_log.append(f"🎯 Removed {removed} rows with missing target values")
        
        # Remove columns with excessive missing values (>90%)
        missing_pct = (df.isnull().sum() / len(df)) * 100
        excessive_missing = missing_pct[missing_pct > 90].index.tolist()
        
        if excessive_missing:
            df = df.drop(columns=excessive_missing)
            self.cleaning_log.append(f"🗑️ Dropped {len(excessive_missing)} columns with >90% missing data: {excessive_missing}")
        
        # Handle duplicate records
        initial_len = len(df)
        df = df.drop_duplicates(keep='first')
        removed_dups = initial_len - len(df)
        if removed_dups > 0:
            self.cleaning_log.append(f"🔄 Removed {removed_dups} duplicate records")
        
        return df
    
    def _ml_missing_value_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced missing value imputation optimized for ML performance."""
        
        for col in df.columns:
            if col == self.target_column:
                continue  # Skip target column
                
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            missing_pct = (missing_count / len(df)) * 100
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Numeric columns - choose strategy based on distribution and missingness
                if missing_pct < 5:
                    # Low missingness - use mean/median based on skewness
                    skewness = abs(df[col].skew())
                    if skewness > 1:
                        fill_value = df[col].median()
                        method = "median (skewed distribution)"
                    else:
                        fill_value = df[col].mean()
                        method = "mean (normal distribution)"
                    df[col] = df[col].fillna(fill_value)
                    
                elif missing_pct < 20:
                    # Medium missingness - use KNN imputation for better accuracy
                    try:
                        # Use other numeric columns for KNN imputation
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if len(numeric_cols) > 1:
                            imputer = KNNImputer(n_neighbors=5)
                            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                            method = "KNN imputation (k=5)"
                        else:
                            df[col] = df[col].fillna(df[col].median())
                            method = "median (fallback)"
                    except:
                        df[col] = df[col].fillna(df[col].median())
                        method = "median (fallback)"
                else:
                    # High missingness - create missing indicator and fill
                    df[f'{col}_was_missing'] = df[col].isnull().astype(int)
                    df[col] = df[col].fillna(df[col].median())
                    method = "median + missing indicator"
                    self.ml_recommendations.append(f"Consider feature engineering for '{col}' - high missingness may be informative")
            
            else:
                # Categorical columns
                if missing_pct < 10:
                    # Low missingness - use mode
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col] = df[col].fillna(mode_value.iloc[0])
                        method = "mode"
                    else:
                        df[col] = df[col].fillna('Unknown')
                        method = "unknown (no mode)"
                else:
                    # High missingness - explicit missing category
                    df[col] = df[col].fillna('Missing_Data')
                    method = "explicit missing category"
            
            self.cleaning_log.append(f"🔧 Filled {missing_count} missing values in '{col}' using {method}")
        
        return df
    
    def _ml_outlier_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """ML-focused outlier detection and treatment."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)  # Don't treat target outliers
        
        for col in numeric_cols:
            # Skip binary/boolean columns
            unique_values = df[col].nunique()
            if unique_values <= 2:
                continue
            
            # Detect outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # No variation
                continue
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_pct = (outlier_count / len(df)) * 100
                
                if outlier_pct < 5:
                    # Few outliers - cap them
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    method = "capping"
                elif outlier_pct < 15:
                    # Moderate outliers - transform
                    # Apply log transformation to reduce impact
                    if df[col].min() > 0:
                        df[col] = np.log1p(df[col])
                        method = "log transformation"
                    else:
                        # Shift then log
                        shift_value = abs(df[col].min()) + 1
                        df[col] = np.log1p(df[col] + shift_value)
                        method = "shifted log transformation"
                else:
                    # Many outliers - create outlier indicator
                    df[f'{col}_is_outlier'] = outliers.astype(int)
                    # Still cap extreme values
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    method = "capping + outlier indicator"
                
                self.cleaning_log.append(f"🎯 Handled {outlier_count} outliers in '{col}' using {method}")
        
        return df
    
    def _ml_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML-friendly features from existing data."""
        
        # Handle datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Extract useful date features
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_quarter'] = df[col].dt.quarter
                
                # Calculate age from date if it looks like a birth date
                if 'birth' in col.lower() or 'dob' in col.lower():
                    df[f'{col}_age'] = (pd.Timestamp.now() - df[col]).dt.days / 365.25
                
                # Drop original datetime column (usually not ML-friendly)
                df = df.drop(columns=[col])
                self.cleaning_log.append(f"📅 Extracted date features from '{col}' and removed original")
        
        # Handle high-cardinality categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            if col == self.target_column:
                continue
                
            unique_count = df[col].nunique()
            total_count = len(df)
            
            # If too many unique values, it might need special handling
            if unique_count > 50 and unique_count > total_count * 0.1:
                # High cardinality - might need grouping
                value_counts = df[col].value_counts()
                
                # Keep top categories, group rest as 'Other'
                top_categories = value_counts.head(20).index.tolist()
                df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
                
                self.cleaning_log.append(f"🏷️ Grouped rare categories in '{col}' (kept top 20, rest as 'Other')")
                self.ml_recommendations.append(f"Consider target encoding for '{col}' if it's informative")
        
        return df
    
    def _optimize_for_ml(self, df: pd.DataFrame) -> tuple:
        """Optimize data types and encode categorical variables for ML."""
        encoders = {}
        
        # Handle categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            if col == self.target_column:
                continue
                
            # Label encode categorical variables
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
            self.cleaning_log.append(f"🔤 Label encoded '{col}' ({len(le.classes_)} categories)")
        
        # Optimize numeric data types (but keep them Streamlit-compatible)
        for col in df.select_dtypes(include=[np.number]).columns:
            # Check if we can downcast to save memory while maintaining compatibility
            if df[col].dtype == 'float64':
                if df[col].isnull().sum() == 0:  # No missing values
                    min_val, max_val = df[col].min(), df[col].max()
                    # Only downcast if significant memory savings and values fit in float32 range
                    if (min_val >= -3.4e38 and max_val <= 3.4e38 and 
                        (max_val - min_val) < 1e6):  # Conservative downcasting
                        # Keep as float64 for Streamlit compatibility
                        pass  # Skip downcasting to avoid Arrow issues
            
            elif df[col].dtype == 'int64':
                min_val, max_val = df[col].min(), df[col].max()
                # Only downcast integers if very large dataset and significant savings
                if (len(df) > 10000 and 
                    min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max):
                    # Keep as int64 for Streamlit compatibility  
                    pass  # Skip downcasting to avoid Arrow issues
        
        self.cleaning_log.append("💾 Maintained data types for optimal Streamlit compatibility")
        
        return df, encoders
    
    def _validate_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and filter features for ML readiness."""
        
        # Remove constant features (no variation)
        constant_features = []
        for col in df.columns:
            if col == self.target_column:
                continue
            if df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            df = df.drop(columns=constant_features)
            self.cleaning_log.append(f"🚫 Removed {len(constant_features)} constant features: {constant_features}")
        
        # Remove features with very low variance
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
        
        if numeric_cols:
            selector = VarianceThreshold(threshold=0.01)  # Remove features with variance < 0.01
            try:
                selected = selector.fit_transform(df[numeric_cols])
                selected_features = selector.get_support(indices=True)
                removed_features = [numeric_cols[i] for i in range(len(numeric_cols)) if i not in selected_features]
                
                if removed_features:
                    df = df.drop(columns=removed_features)
                    self.cleaning_log.append(f"📉 Removed {len(removed_features)} low-variance features: {removed_features}")
            except:
                pass  # Skip if transformation fails
        
        return df
    
    def _generate_ml_readiness_report(self, df: pd.DataFrame) -> dict:
        """Generate comprehensive ML readiness assessment."""
        
        report = {
            'data_shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'ml_readiness_score': 0
        }
        
        # Calculate ML readiness score (0-100)
        score = 100
        
        # Penalize missing values
        missing_pct = (report['missing_values'] / (df.shape[0] * df.shape[1])) * 100
        score -= min(missing_pct * 2, 30)  # Max 30 point penalty
        
        # Penalize duplicates
        duplicate_pct = (report['duplicate_rows'] / df.shape[0]) * 100
        score -= min(duplicate_pct, 20)  # Max 20 point penalty
        
        # Bonus for good data types
        if report['categorical_features'] == 0:  # All numeric
            score += 10
        
        # Bonus for reasonable size
        if 1000 <= df.shape[0] <= 100000:  # Good sample size
            score += 5
        
        report['ml_readiness_score'] = max(0, min(100, score))
        
        # Readiness assessment
        if report['ml_readiness_score'] >= 90:
            report['readiness_level'] = 'Excellent'
            report['readiness_color'] = 'green'
        elif report['ml_readiness_score'] >= 75:
            report['readiness_level'] = 'Good'
            report['readiness_color'] = 'blue'
        elif report['ml_readiness_score'] >= 60:
            report['readiness_level'] = 'Fair'
            report['readiness_color'] = 'orange'
        else:
            report['readiness_level'] = 'Poor'
            report['readiness_color'] = 'red'
        
        return report
