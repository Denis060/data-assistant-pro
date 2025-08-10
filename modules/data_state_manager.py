"""
Data State Validation and Consistency Manager
Ensures all data references across the system are consistent and valid
"""

import logging
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DataStateManager:
    """Centralized data state management for consistency across the application."""
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
        
    def validate_session_state(self) -> Dict[str, Any]:
        """Comprehensive validation of session state data consistency."""
        validation_report = {
            'status': 'healthy',
            'errors': [],
            'warnings': [],
            'data_summary': {},
            'recommendations': []
        }
        
        try:
            # Check data existence
            data_sources = self._check_data_sources()
            validation_report['data_summary'] = data_sources
            
            # Validate data consistency
            consistency_issues = self._validate_data_consistency()
            validation_report['errors'].extend(consistency_issues['errors'])
            validation_report['warnings'].extend(consistency_issues['warnings'])
            
            # Check data integrity
            integrity_issues = self._validate_data_integrity()
            validation_report['errors'].extend(integrity_issues['errors'])
            validation_report['warnings'].extend(integrity_issues['warnings'])
            
            # Generate recommendations
            validation_report['recommendations'] = self._generate_recommendations(validation_report)
            
            # Set overall status
            if validation_report['errors']:
                validation_report['status'] = 'error'
            elif validation_report['warnings']:
                validation_report['status'] = 'warning'
                
        except Exception as e:
            validation_report['status'] = 'critical_error'
            validation_report['errors'].append(f"Validation system error: {str(e)}")
            logger.error(f"Data state validation failed: {str(e)}")
            
        return validation_report
    
    def _check_data_sources(self) -> Dict[str, Any]:
        """Check what data sources are available in session state."""
        sources = {}
        
        # Standard data variables
        data_vars = ['cleaned_df', 'original_df', 'sample_data', 'df']
        
        for var in data_vars:
            if var in st.session_state:
                data = st.session_state[var]
                if isinstance(data, pd.DataFrame):
                    sources[var] = {
                        'exists': True,
                        'shape': data.shape,
                        'memory_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                        'missing_values': data.isnull().sum().sum(),
                        'dtypes': data.dtypes.to_dict()
                    }
                else:
                    sources[var] = {'exists': True, 'type': type(data).__name__, 'valid': False}
            else:
                sources[var] = {'exists': False}
                
        return sources
    
    def _validate_data_consistency(self) -> Dict[str, List[str]]:
        """Validate consistency between different data sources."""
        issues = {'errors': [], 'warnings': []}
        
        # Check if multiple data sources exist and are consistent
        active_data = {}
        for var in ['cleaned_df', 'original_df', 'sample_data']:
            if var in st.session_state and isinstance(st.session_state[var], pd.DataFrame):
                active_data[var] = st.session_state[var]
        
        if len(active_data) > 1:
            # Compare shapes and columns
            shapes = {k: v.shape for k, v in active_data.items()}
            columns = {k: list(v.columns) for k, v in active_data.items()}
            
            # Check if original_df and cleaned_df have same columns
            if 'original_df' in active_data and 'cleaned_df' in active_data:
                orig_cols = set(active_data['original_df'].columns)
                clean_cols = set(active_data['cleaned_df'].columns)
                
                if orig_cols != clean_cols:
                    missing_cols = orig_cols - clean_cols
                    added_cols = clean_cols - orig_cols
                    
                    if missing_cols:
                        issues['warnings'].append(f"Columns removed during cleaning: {list(missing_cols)}")
                    if added_cols:
                        issues['warnings'].append(f"Columns added during cleaning: {list(added_cols)}")
            
            # Check for data hash consistency
            if 'data_hash' in st.session_state:
                from .cache_utils import DataCache
                if 'cleaned_df' in active_data:
                    current_hash = DataCache.get_data_hash(active_data['cleaned_df'])
                    if current_hash != st.session_state.get('data_hash'):
                        issues['warnings'].append("Data hash mismatch - cache may be stale")
        
        elif len(active_data) == 0:
            issues['errors'].append("No valid data sources found in session state")
            
        return issues
    
    def _validate_data_integrity(self) -> Dict[str, List[str]]:
        """Validate data integrity for active datasets."""
        issues = {'errors': [], 'warnings': []}
        
        for var_name in ['cleaned_df', 'original_df', 'sample_data']:
            if var_name in st.session_state:
                data = st.session_state[var_name]
                if isinstance(data, pd.DataFrame):
                    # Check for completely empty dataframe
                    if data.empty:
                        issues['errors'].append(f"{var_name} is empty")
                        continue
                    
                    # Check for suspicious data patterns
                    if data.shape[0] < 2:
                        issues['warnings'].append(f"{var_name} has only {data.shape[0]} row(s)")
                    
                    if data.shape[1] < 1:
                        issues['errors'].append(f"{var_name} has no columns")
                    
                    # Check for data types
                    if data.dtypes.apply(lambda x: x == 'object').all():
                        issues['warnings'].append(f"{var_name} contains only object columns - may need type conversion")
                    
                    # Check for extreme missing values
                    missing_pct = (data.isnull().sum().sum() / data.size) * 100
                    if missing_pct > 90:
                        issues['warnings'].append(f"{var_name} has {missing_pct:.1f}% missing values")
                    elif missing_pct > 99:
                        issues['errors'].append(f"{var_name} has {missing_pct:.1f}% missing values")
        
        return issues
    
    def _generate_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Data source recommendations
        data_sources = validation_report['data_summary']
        active_sources = [k for k, v in data_sources.items() if v.get('exists') and v.get('shape')]
        
        if len(active_sources) == 0:
            recommendations.append("Load data using file upload or sample datasets")
        elif len(active_sources) > 2:
            recommendations.append("Consider consolidating data sources for clarity")
        
        # Error-based recommendations
        if any("No valid data sources" in error for error in validation_report['errors']):
            recommendations.append("Initialize data by uploading a file or loading sample data")
        
        if any("missing values" in warning for warning in validation_report['warnings']):
            recommendations.append("Consider handling missing values using the cleaning tools")
        
        if any("hash mismatch" in warning for warning in validation_report['warnings']):
            recommendations.append("Clear cache or reload data to ensure consistency")
        
        return recommendations
    
    def fix_common_issues(self) -> List[str]:
        """Automatically fix common data state issues."""
        fixes_applied = []
        
        try:
            # Fix missing data hash
            if 'cleaned_df' in st.session_state and 'data_hash' not in st.session_state:
                from .cache_utils import DataCache
                st.session_state.data_hash = DataCache.get_data_hash(st.session_state.cleaned_df)
                fixes_applied.append("Added missing data hash")
            
            # Ensure original_df exists if cleaned_df exists
            if 'cleaned_df' in st.session_state and 'original_df' not in st.session_state:
                st.session_state.original_df = st.session_state.cleaned_df.copy()
                fixes_applied.append("Created original_df backup")
            
            # Remove invalid data references
            for var in ['cleaned_df', 'original_df', 'sample_data']:
                if var in st.session_state:
                    data = st.session_state[var]
                    if not isinstance(data, pd.DataFrame) or data.empty:
                        del st.session_state[var]
                        fixes_applied.append(f"Removed invalid {var}")
                        
        except Exception as e:
            logger.error(f"Error applying automatic fixes: {str(e)}")
            fixes_applied.append(f"Error in auto-fix: {str(e)}")
        
        return fixes_applied

def validate_data_state() -> Dict[str, Any]:
    """Quick validation function for use throughout the application."""
    manager = DataStateManager()
    return manager.validate_session_state()

def auto_fix_data_state() -> List[str]:
    """Quick auto-fix function for use throughout the application."""
    manager = DataStateManager()
    return manager.fix_common_issues()

def data_state_dashboard():
    """Streamlit dashboard for data state management."""
    st.subheader("🔍 Data State Health Check")
    
    # Run validation
    with st.spinner("Validating data state..."):
        validation_report = validate_data_state()
    
    # Display status
    status = validation_report['status']
    if status == 'healthy':
        st.success("✅ Data state is healthy!")
    elif status == 'warning':
        st.warning("⚠️ Data state has warnings")
    elif status == 'error':
        st.error("❌ Data state has errors")
    else:
        st.error("🚨 Critical data state error")
    
    # Show details in expandable sections
    if validation_report['data_summary']:
        with st.expander("📊 Data Sources Summary"):
            for source, info in validation_report['data_summary'].items():
                if info.get('exists') and info.get('shape'):
                    st.write(f"**{source}**: {info['shape']} | {info['memory_mb']:.1f}MB | {info['missing_values']} missing")
                elif info.get('exists'):
                    st.write(f"**{source}**: {info.get('type', 'Unknown')} (invalid)")
                else:
                    st.write(f"**{source}**: Not found")
    
    if validation_report['errors']:
        with st.expander("❌ Errors", expanded=True):
            for error in validation_report['errors']:
                st.error(error)
    
    if validation_report['warnings']:
        with st.expander("⚠️ Warnings"):
            for warning in validation_report['warnings']:
                st.warning(warning)
    
    if validation_report['recommendations']:
        with st.expander("💡 Recommendations"):
            for rec in validation_report['recommendations']:
                st.info(f"• {rec}")
    
    # Auto-fix button
    if validation_report['errors'] or validation_report['warnings']:
        if st.button("🔧 Auto-Fix Common Issues"):
            with st.spinner("Applying fixes..."):
                fixes = auto_fix_data_state()
            
            if fixes:
                st.success("Applied fixes:")
                for fix in fixes:
                    st.write(f"✅ {fix}")
                st.rerun()
            else:
                st.info("No automatic fixes available")
