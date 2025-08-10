"""
Advanced caching utilities for Data Assistant Pro
Provides memory and disk-based caching for expensive operations
"""

import hashlib
import pickle
import os
import time
from typing import Any, Callable, Optional
import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Create cache directory
CACHE_DIR = ".cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

class DataCache:
    """Advanced caching system for expensive data operations"""
    
    @staticmethod
    def get_data_hash(df: pd.DataFrame) -> str:
        """Generate a hash for a DataFrame for caching purposes"""
        try:
            # Create hash based on data shape, columns, and sample of data
            content = f"{df.shape}_{df.columns.tolist()}_{df.head().to_string()}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not generate data hash: {e}")
            return str(time.time())
    
    @staticmethod
    def cache_operation(func: Callable, cache_key: str, ttl_hours: int = 24) -> Any:
        """
        Cache expensive operations to disk
        
        Args:
            func: Function to cache
            cache_key: Unique key for this operation
            ttl_hours: Time to live in hours
        
        Returns:
            Cached result or fresh computation
        """
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
        
        # Check if cache exists and is still valid
        if os.path.exists(cache_file):
            try:
                # Check file age
                file_age = time.time() - os.path.getmtime(cache_file)
                if file_age < ttl_hours * 3600:  # Convert hours to seconds
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                        logger.info(f"Using cached result for {cache_key}")
                        return result
                else:
                    logger.info(f"Cache expired for {cache_key}, recomputing...")
            except Exception as e:
                # Smart error handling for cache corruption
                from .error_handler_v2 import error_handler, display_smart_error
                
                context = {
                    'cache_key': cache_key,
                    'cache_file': cache_file,
                    'operation': 'cache_loading'
                }
                
                smart_error = error_handler.analyze_error(e, context)
                action = display_smart_error(smart_error)
                
                if action == "clear_cache":
                    try:
                        import os
                        os.remove(cache_file)
                        st.success("🧹 Corrupted cache file removed")
                    except:
                        pass
                elif action == "retry_operation":
                    # Will fall through to recomputation
                    pass
                
                logger.warning(f"Cache loading failed for {cache_key}: {e}")
        
        # Compute fresh result
        logger.info(f"Computing fresh result for {cache_key}")
        result = func()
        
        # Save to cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            logger.info(f"Cached result for {cache_key}")
        except Exception as e:
            logger.warning(f"Could not save cache for {cache_key}: {e}")
        
        return result
    
    @staticmethod
    def clear_cache():
        """Clear all cached files"""
        try:
            import glob
            cache_files = glob.glob(os.path.join(CACHE_DIR, "*.pkl"))
            for file in cache_files:
                os.remove(file)
            logger.info(f"Cleared {len(cache_files)} cache files")
            return len(cache_files)
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    @staticmethod
    def get_cache_info():
        """Get information about cached files"""
        try:
            import glob
            cache_files = glob.glob(os.path.join(CACHE_DIR, "*.pkl"))
            total_size = sum(os.path.getsize(f) for f in cache_files)
            return {
                "file_count": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "files": [
                    {
                        "name": os.path.basename(f),
                        "size_mb": os.path.getsize(f) / (1024 * 1024),
                        "age_hours": (time.time() - os.path.getmtime(f)) / 3600
                    }
                    for f in cache_files
                ]
            }
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {"file_count": 0, "total_size_mb": 0, "files": []}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_correlation_matrix(df_hash: str, df: pd.DataFrame):
    """Cached correlation matrix computation"""
    logger.info("Computing correlation matrix...")
    return df.select_dtypes(include=['number']).corr()

@st.cache_data(ttl=3600)  # Cache for 1 hour  
def cached_missing_analysis(df_hash: str, df: pd.DataFrame):
    """Cached missing value analysis"""
    try:
        logger.info("Computing missing value analysis...")
        missing_count = df.isnull().sum()
        missing_percentage = (missing_count / len(df)) * 100
        
        missing_info = {
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
            'total_missing': missing_count.sum(),
            'columns_with_missing': df.columns[df.isnull().any()].tolist()
        }
        return missing_info
    except Exception as e:
        logger.error(f"Error in missing analysis: {e}")
        # Return safe fallback
        return {
            'missing_count': pd.Series(0, index=df.columns),
            'missing_percentage': pd.Series(0.0, index=df.columns),
            'total_missing': 0,
            'columns_with_missing': []
        }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_statistical_summary(df_hash: str, df: pd.DataFrame):
    """Cached statistical summary"""
    try:
        logger.info("Computing statistical summary...")
        numeric_df = df.select_dtypes(include=['number'])
        
        return {
            'describe': numeric_df.describe(),
            'dtypes': df.dtypes,
            'memory_usage': df.memory_usage(deep=True),
            'nunique': df.nunique()
        }
    except Exception as e:
        logger.error(f"Error in statistical summary: {e}")
        # Return safe fallback
        return {
            'describe': pd.DataFrame(),
            'dtypes': df.dtypes,
            'memory_usage': pd.Series(0, index=df.columns),
            'nunique': pd.Series(1, index=df.columns)
        }

def with_progress_cache(operation_name: str, func: Callable, *args, **kwargs):
    """
    Execute a function with progress indication and caching
    
    Args:
        operation_name: Name to display in progress
        func: Function to execute
        *args, **kwargs: Arguments for the function
    
    Returns:
        Function result
    """
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(f"🔄 {operation_name}...")
        progress_bar.progress(25)
        
        # Execute function
        result = func(*args, **kwargs)
        
        progress_bar.progress(75)
        status_text.text(f"✅ {operation_name} complete!")
        progress_bar.progress(100)
        
        time.sleep(0.3)  # Brief pause for visual feedback
        
        # Clean up
        progress_bar.empty()
        status_text.empty()
        
        return result
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error in {operation_name}: {str(e)}")
        raise e
