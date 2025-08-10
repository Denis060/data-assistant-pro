"""
Domain-Specific Data Validation Rules
Catch business logic errors and domain-specific data quality issues
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Any
import re
from datetime import datetime, date


class DataValidationRules:
    """Domain-specific validation rules for common data scenarios."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validation_errors = []
        self.warnings = []
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        
        # Business logic validations
        self._validate_demographics()
        self._validate_financial_data()
        self._validate_dates()
        self._validate_identifiers()
        self._validate_measurements()
        self._validate_categorical_data()
        self._validate_relationships()
        
        return {
            'errors': self.validation_errors,
            'warnings': self.warnings,
            'total_issues': len(self.validation_errors) + len(self.warnings)
        }
    
    def _validate_demographics(self):
        """Validate demographic data (age, gender, etc.)."""
        
        # Age validation
        age_cols = [col for col in self.df.columns if 'age' in col.lower()]
        for col in age_cols:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check for impossible ages
                too_young = (self.df[col] < 0).sum()
                too_old = (self.df[col] > 150).sum()
                
                if too_young > 0:
                    self.validation_errors.append(f"'{col}': {too_young} records with negative age")
                if too_old > 0:
                    self.validation_errors.append(f"'{col}': {too_old} records with age > 150")
                
                # Check for unusual age distributions
                if self.df[col].median() < 5:
                    self.warnings.append(f"'{col}': Unusually young population (median age: {self.df[col].median():.1f})")
        
        # Gender validation
        gender_cols = [col for col in self.df.columns if any(term in col.lower() for term in ['gender', 'sex'])]
        for col in gender_cols:
            unique_values = self.df[col].dropna().unique()
            expected_values = ['male', 'female', 'm', 'f', 'other', 'non-binary', 'prefer not to say']
            
            unexpected = [val for val in unique_values if str(val).lower() not in expected_values]
            if unexpected:
                self.warnings.append(f"'{col}': Unexpected gender values: {unexpected}")
    
    def _validate_financial_data(self):
        """Validate financial data (salary, price, revenue, etc.)."""
        
        financial_cols = [col for col in self.df.columns 
                         if any(term in col.lower() for term in 
                               ['salary', 'income', 'wage', 'price', 'cost', 'revenue', 'amount', 'value'])]
        
        for col in financial_cols:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Check for negative financial values (usually impossible)
                negative_count = (self.df[col] < 0).sum()
                if negative_count > 0:
                    self.validation_errors.append(f"'{col}': {negative_count} records with negative values")
                
                # Check for unrealistic values
                if 'salary' in col.lower() or 'income' in col.lower():
                    extremely_high = (self.df[col] > 10000000).sum()  # > $10M salary
                    extremely_low = ((self.df[col] > 0) & (self.df[col] < 1000)).sum()  # < $1K salary
                    
                    if extremely_high > 0:
                        self.warnings.append(f"'{col}': {extremely_high} records with extremely high values (>$10M)")
                    if extremely_low > 0:
                        self.warnings.append(f"'{col}': {extremely_low} records with extremely low values (<$1K)")
    
    def _validate_dates(self):
        """Validate date columns for logical consistency."""
        
        date_cols = []
        for col in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                date_cols.append(col)
            elif any(term in col.lower() for term in ['date', 'time', 'created', 'updated', 'birth']):
                date_cols.append(col)
        
        for col in date_cols:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    date_series = pd.to_datetime(self.df[col], errors='coerce')
                else:
                    date_series = self.df[col]
                
                # Check for future dates where they shouldn't exist
                if any(term in col.lower() for term in ['birth', 'hire', 'start', 'created']):
                    future_dates = (date_series > datetime.now()).sum()
                    if future_dates > 0:
                        self.validation_errors.append(f"'{col}': {future_dates} records with future dates")
                
                # Check for very old dates
                very_old = (date_series < datetime(1900, 1, 1)).sum()
                if very_old > 0:
                    self.warnings.append(f"'{col}': {very_old} records with dates before 1900")
                
            except Exception:
                self.warnings.append(f"'{col}': Unable to validate date format")
        
        # Check date relationships
        self._validate_date_relationships(date_cols)
    
    def _validate_date_relationships(self, date_cols: List[str]):
        """Validate logical relationships between date columns."""
        
        # Common date relationship patterns
        date_patterns = [
            (['birth', 'dob'], ['hire', 'start', 'join']),
            (['start', 'begin', 'open'], ['end', 'close', 'finish']),
            (['created', 'issued'], ['updated', 'modified']),
        ]
        
        for col1_terms, col2_terms in date_patterns:
            col1_matches = [col for col in date_cols if any(term in col.lower() for term in col1_terms)]
            col2_matches = [col for col in date_cols if any(term in col.lower() for term in col2_terms)]
            
            for col1 in col1_matches:
                for col2 in col2_matches:
                    if col1 != col2:
                        try:
                            date1 = pd.to_datetime(self.df[col1], errors='coerce')
                            date2 = pd.to_datetime(self.df[col2], errors='coerce')
                            
                            # Check if col1 should be before col2
                            invalid_order = (date1 > date2).sum()
                            if invalid_order > 0:
                                self.validation_errors.append(
                                    f"Date order violation: {invalid_order} records where '{col1}' > '{col2}'"
                                )
                        except Exception:
                            pass
    
    def _validate_identifiers(self):
        """Validate ID columns and unique identifiers."""
        
        id_cols = [col for col in self.df.columns 
                  if any(term in col.lower() for term in ['id', '_key', 'uuid', 'identifier'])]
        
        for col in id_cols:
            # Check for duplicates in ID columns
            duplicates = self.df[col].duplicated().sum()
            if duplicates > 0:
                self.validation_errors.append(f"'{col}': {duplicates} duplicate values in ID column")
            
            # Check for missing values in ID columns
            missing = self.df[col].isnull().sum()
            if missing > 0:
                self.validation_errors.append(f"'{col}': {missing} missing values in ID column")
            
            # Check ID format consistency
            if self.df[col].dtype == 'object':
                # Check if IDs follow a consistent pattern
                str_lengths = self.df[col].astype(str).str.len()
                if str_lengths.nunique() > 3:  # More than 3 different lengths
                    self.warnings.append(f"'{col}': Inconsistent ID format (varying lengths)")
    
    def _validate_measurements(self):
        """Validate measurement data (height, weight, distance, etc.)."""
        
        measurement_patterns = {
            'height': (0.3, 3.0),    # meters
            'weight': (0.5, 500),    # kg
            'distance': (0, 50000),  # km
            'temperature': (-50, 60), # celsius
            'speed': (0, 300),       # km/h
        }
        
        for col in self.df.columns:
            for pattern, (min_val, max_val) in measurement_patterns.items():
                if pattern in col.lower() and pd.api.types.is_numeric_dtype(self.df[col]):
                    too_low = (self.df[col] < min_val).sum()
                    too_high = (self.df[col] > max_val).sum()
                    
                    if too_low > 0:
                        self.validation_errors.append(
                            f"'{col}': {too_low} records below reasonable {pattern} range"
                        )
                    if too_high > 0:
                        self.validation_errors.append(
                            f"'{col}': {too_high} records above reasonable {pattern} range"
                        )
    
    def _validate_categorical_data(self):
        """Validate categorical data for consistency."""
        
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            
            # Check for too many categories (might indicate data quality issues)
            if unique_count > total_count * 0.8:
                self.warnings.append(
                    f"'{col}': Very high cardinality ({unique_count} unique values), might be an ID column"
                )
            
            # Check for potential data entry inconsistencies
            if unique_count < 50:  # Only check for smaller categorical sets
                values = self.df[col].dropna().astype(str)
                
                # Check for similar values that might be typos
                value_counts = values.value_counts()
                for val in value_counts.index:
                    similar_vals = [v for v in value_counts.index 
                                  if v != val and self._strings_similar(val, v)]
                    if similar_vals:
                        self.warnings.append(
                            f"'{col}': Possible typos - '{val}' similar to {similar_vals}"
                        )
    
    def _validate_relationships(self):
        """Validate logical relationships between columns."""
        
        # Check for sum relationships (parts should sum to total)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Look for columns that might be parts of a total
        potential_totals = [col for col in numeric_cols if 'total' in col.lower()]
        
        for total_col in potential_totals:
            # Find potential component columns
            base_name = total_col.lower().replace('total', '').replace('_', '').replace(' ', '')
            component_cols = [col for col in numeric_cols 
                            if col != total_col and base_name in col.lower()]
            
            if len(component_cols) >= 2:
                # Check if components sum to total
                calculated_total = self.df[component_cols].sum(axis=1)
                actual_total = self.df[total_col]
                
                # Allow for small rounding differences
                diff_threshold = 0.01 * actual_total.mean() if actual_total.mean() > 0 else 0.01
                significant_diffs = (np.abs(calculated_total - actual_total) > diff_threshold).sum()
                
                if significant_diffs > 0:
                    self.validation_errors.append(
                        f"Sum validation: {significant_diffs} records where components don't sum to '{total_col}'"
                    )
    
    def _strings_similar(self, s1: str, s2: str, threshold: float = 0.8) -> bool:
        """Check if two strings are similar (simple implementation)."""
        if len(s1) == 0 or len(s2) == 0:
            return False
        
        # Simple similarity check based on character overlap
        set1, set2 = set(s1.lower()), set(s2.lower())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union > threshold if union > 0 else False


def domain_validation_dashboard(df: pd.DataFrame) -> None:
    """Dashboard for domain-specific data validation."""
    
    st.subheader("🎯 Domain-Specific Validation")
    st.write("Checking for business logic errors and domain-specific issues...")
    
    # Run validation
    validator = DataValidationRules(df)
    validation_results = validator.validate_all()
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        error_count = len(validation_results['errors'])
        if error_count == 0:
            st.success(f"✅ No Errors Found")
        else:
            st.error(f"❌ {error_count} Errors Found")
    
    with col2:
        warning_count = len(validation_results['warnings'])
        if warning_count == 0:
            st.success(f"✅ No Warnings")
        else:
            st.warning(f"⚠️ {warning_count} Warnings")
    
    with col3:
        total_issues = validation_results['total_issues']
        if total_issues == 0:
            st.success("🎉 All Validations Passed")
        else:
            st.info(f"📋 {total_issues} Total Issues")
    
    # Show detailed results
    if validation_results['errors']:
        st.subheader("🚨 Critical Errors")
        for i, error in enumerate(validation_results['errors'], 1):
            st.error(f"{i}. {error}")
    
    if validation_results['warnings']:
        st.subheader("⚠️ Warnings")
        for i, warning in enumerate(validation_results['warnings'], 1):
            st.warning(f"{i}. {warning}")
    
    if not validation_results['errors'] and not validation_results['warnings']:
        st.success("🎉 Your data passed all domain-specific validation checks!")
        st.balloons()
