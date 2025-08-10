"""Enhanced data loading utilities for Data Assistant Pro."""

import io
import logging
import time
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
from .error_handler_v2 import (
    handle_error, 
    ErrorSeverity, 
    ErrorCategory,
    StandardizedError
)

logger = logging.getLogger(__name__)


def detect_delimiter(file_content: str) -> str:
    """Detect the delimiter used in a CSV file.

    Args:
        file_content: String content of the file

    Returns:
        The best detected delimiter
    """
    import csv

    # Try different delimiters
    delimiters = [",", ";", "\t", "|"]

    # Read first few lines to detect delimiter
    lines = file_content.split("\n")[:5]  # Check first 5 lines

    delimiter_scores = {}

    for delimiter in delimiters:
        try:
            # Count how many fields we get with this delimiter
            field_counts = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    field_count = len(line.split(delimiter))
                    field_counts.append(field_count)

            if field_counts:
                # Score based on consistency of field counts
                avg_fields = sum(field_counts) / len(field_counts)
                consistency = 1 - (max(field_counts) - min(field_counts)) / max(
                    field_counts, 1
                )
                score = avg_fields * consistency
                delimiter_scores[delimiter] = score
        except Exception:
            delimiter_scores[delimiter] = 0

    # Return the delimiter with the highest score
    if delimiter_scores:
        best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
        return best_delimiter
    return ","


def smart_read_csv(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """Smart CSV reading with automatic delimiter detection.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Tuple of (DataFrame, delimiter_used)
    """
    try:
        # Read file content for delimiter detection
        uploaded_file.seek(0)
        content = uploaded_file.read().decode("utf-8")

        # Detect delimiter
        delimiter = detect_delimiter(content)

        # Reset file pointer and read with detected delimiter
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=delimiter)

        logger.info(f"Successfully loaded CSV with delimiter '{delimiter}': {df.shape}")
        return df, delimiter

    except Exception as e:
        logger.error(f"Error in smart_read_csv: {str(e)}")
        # Fallback to standard comma-separated reading
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            return df, ","
        except Exception as fallback_error:
            logger.error(f"Fallback CSV reading failed: {str(fallback_error)}")
            return None, ","


@st.cache_data
def load_data(
    uploaded_file, delimiter_override: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """Enhanced data loading with comprehensive validation and smart delimiter detection.

    Args:
        uploaded_file: Streamlit uploaded file object
        delimiter_override: Manual delimiter selection (optional)

    Returns:
        Loaded DataFrame or None if loading fails
    """
    if uploaded_file is not None:
        try:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Log file upload
            logger.info(
                f"Loading file: {uploaded_file.name}, Size: {uploaded_file.size} bytes"
            )
            
            # Step 1: File validation
            status_text.text("🔍 Validating file...")
            progress_bar.progress(10)
            time.sleep(0.1)  # Small delay for visual feedback

            # File size validation (max 100MB)
            if uploaded_file.size > 100 * 1024 * 1024:
                progress_bar.empty()
                status_text.empty()
                st.error("❌ File too large! Please upload a file smaller than 100MB.")
                return None

            # Step 2: Determine delimiter
            status_text.text("🔧 Detecting file format...")
            progress_bar.progress(25)
            
            # Determine delimiter to use
            if delimiter_override and delimiter_override != "Auto-detect":
                # Use manual delimiter selection
                delimiter_map = {
                    "Comma (,)": ",",
                    "Semicolon (;)": ";",
                    "Tab": "\t",
                    "Pipe (|)": "|",
                }
                delimiter = delimiter_map.get(delimiter_override, ",")
                delimiter_source = "manually selected"
            else:
                # Auto-detect delimiter
                status_text.text("🤖 Auto-detecting delimiter...")
                progress_bar.progress(40)
                delimiter_source = "auto-detected"
            
            # Step 3: Load data
            status_text.text("📊 Loading data...")
            progress_bar.progress(60)
            
            if delimiter_override and delimiter_override != "Auto-detect":
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=delimiter)
            else:
                # Use smart CSV reading with delimiter detection
                df, delimiter = smart_read_csv(uploaded_file)

            if df is None:
                progress_bar.empty()
                status_text.empty()
                return None
                
            # Step 4: Data validation
            status_text.text("✅ Validating data structure...")
            progress_bar.progress(80)
            time.sleep(0.2)
            
            # Step 5: Complete
            progress_bar.progress(100)
            status_text.text("🎉 Data loaded successfully!")
            time.sleep(0.5)
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()

            # Show delimiter detection info
            if delimiter != ",":
                if delimiter == "\t":
                    st.success(
                        f"📊 Tab-separated file loaded successfully ({delimiter_source})"
                    )
                elif delimiter == ";":
                    st.success(
                        f"📊 Semicolon-separated file loaded successfully ({delimiter_source})"
                    )
                elif delimiter == "|":
                    st.success(
                        f"📊 Pipe-separated file loaded successfully ({delimiter_source})"
                    )
                else:
                    st.success(
                        f"📊 File loaded with '{delimiter}' delimiter ({delimiter_source})"
                    )
            else:
                st.success(f"📊 CSV file loaded successfully ({delimiter_source})")

            # Basic data validation
            if df.empty:
                st.error("❌ The uploaded file is empty.")
                return None

            if len(df.columns) == 0:
                st.error("❌ No columns found in the file.")
                return None

            if len(df) > 50000:
                st.warning(
                    f"⚠️ Large dataset detected ({len(df):,} rows). Processing may take longer."
                )

            # Log successful load
            logger.info(
                f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns"
            )

            return df

        except pd.errors.EmptyDataError:
            st.error("❌ The file appears to be empty or corrupted.")
            logger.error("Empty data error when loading file")
            return None
        except UnicodeDecodeError as e:
            progress_bar.empty()
            status_text.empty()
            
            # Create context for smart error handling
            context = {
                'file_name': uploaded_file.name,
                'file_size': uploaded_file.size,
                'encoding_attempted': 'utf-8'  # Default encoding
            }
            
            standardized_error = handle_error(
                e, 
                message=f"Failed to load file {uploaded_file.name}",
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.DATA_LOADING,
                suggested_action="Try a different file format or check file encoding"
            )
            
            # Display error to user
            st.error(f"❌ {standardized_error.message}")
            if standardized_error.suggested_action:
                st.info(f"💡 {standardized_error.suggested_action}")
            
            # Return None to indicate failure
            return None
            
        except pd.errors.ParserError as e:
            progress_bar.empty()
            status_text.empty()
            
            context = {
                'file_name': uploaded_file.name,
                'delimiter_used': delimiter,
                'encoding_used': 'utf-8'
            }
            
            standardized_error = handle_error(
                e,
                message=f"Failed to parse CSV file with delimiter '{delimiter}'", 
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.DATA_LOADING,
                suggested_action="Try auto-detecting delimiter or manually specify format"
            )
            
            # Display error to user
            st.error(f"❌ {standardized_error.message}")
            if standardized_error.suggested_action:
                st.info(f"💡 {standardized_error.suggested_action}")
            
            # Return None to indicate failure
            return None
            
        except MemoryError as e:
            progress_bar.empty()
            status_text.empty()
            
            context = {
                'file_name': uploaded_file.name,
                'file_size': uploaded_file.size,
                'data_shape': (None, None)  # Unknown at this point
            }
            
            standardized_error = handle_error(
                e,
                message=f"Memory error loading large file {uploaded_file.name}",
                severity=ErrorSeverity.WARNING,
                category=ErrorCategory.DATA_LOADING,
                suggested_action="Try loading a sample of the data or use a smaller file"
            )
            
            # Display error to user
            st.warning(f"⚠️ {standardized_error.message}")
            if standardized_error.suggested_action:
                st.info(f"💡 {standardized_error.suggested_action}")
            
            # Return None to indicate failure
            return None
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            
            context = {
                'file_name': uploaded_file.name,
                'file_size': uploaded_file.size,
                'operation': 'data_loading'
            }
            
            standardized_error = handle_error(
                e,
                message=f"Unexpected error loading {uploaded_file.name}",
                severity=ErrorSeverity.ERROR, 
                category=ErrorCategory.DATA_LOADING,
                suggested_action="Please check file format and try again"
            )
            
            # Display error to user
            st.error(f"❌ {standardized_error.message}")
            if standardized_error.suggested_action:
                st.info(f"💡 {standardized_error.suggested_action}")
            
            return None
    return None


def validate_data_for_cleaning(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate data before cleaning operations.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, message)
    """
    if df is None or df.empty:
        return False, "No data available for cleaning."

    if len(df.columns) == 0:
        return False, "No columns available for cleaning."

    return True, "Data is valid for cleaning."


def validate_data_for_modeling(
    df: pd.DataFrame, target_column: str
) -> Tuple[bool, str]:
    """Validate data before modeling operations.

    Args:
        df: DataFrame to validate
        target_column: Name of the target column

    Returns:
        Tuple of (is_valid, message)
    """
    if df is None or df.empty:
        return False, "No data available for modeling."

    if target_column not in df.columns:
        return False, f"Target column '{target_column}' not found in the data."

    if len(df.columns) < 2:
        return (
            False,
            "At least 2 columns (1 feature + 1 target) are required for modeling.",
        )

    # Check if target has valid values
    if df[target_column].isnull().all():
        return False, f"Target column '{target_column}' contains no valid values."

    return True, "Data is valid for modeling."


# Quick Fix Helper Functions

def load_data_with_encoding(uploaded_file, delimiter: str, encoding: str) -> Optional[pd.DataFrame]:
    """Load data with specific encoding"""
    try:
        uploaded_file.seek(0)  # Reset file position
        df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding)
        st.success(f"✅ File loaded successfully with {encoding} encoding!")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load with {encoding} encoding: {str(e)}")
        return None

def load_data_with_delimiter(uploaded_file, delimiter: str) -> Optional[pd.DataFrame]:
    """Load data with specific delimiter"""
    try:
        uploaded_file.seek(0)  # Reset file position
        df = pd.read_csv(uploaded_file, delimiter=delimiter)
        st.success(f"✅ File loaded successfully with '{delimiter}' delimiter!")
        return df
    except Exception as e:
        st.error(f"❌ Failed to load with '{delimiter}' delimiter: {str(e)}")
        return None

def retry_with_auto_delimiter(uploaded_file) -> Optional[pd.DataFrame]:
    """Retry loading with automatic delimiter detection"""
    try:
        uploaded_file.seek(0)
        file_content = uploaded_file.read().decode('utf-8')
        detected_delimiter = detect_delimiter(file_content)
        
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, delimiter=detected_delimiter)
        st.success(f"✅ File loaded successfully with auto-detected delimiter: '{detected_delimiter}'")
        return df
    except Exception as e:
        st.error(f"❌ Auto-detection failed: {str(e)}")
        return None

def load_sampled_data(uploaded_file, sample_rate: float = 0.5) -> Optional[pd.DataFrame]:
    """Load a sample of the data to reduce memory usage"""
    try:
        uploaded_file.seek(0)
        
        # First, get the total number of rows
        temp_df = pd.read_csv(uploaded_file, nrows=1000)  # Read first 1000 rows to estimate
        uploaded_file.seek(0)
        
        # Calculate skip rows for sampling
        import random
        total_rows = sum(1 for line in uploaded_file) - 1  # Exclude header
        uploaded_file.seek(0)
        
        rows_to_read = int(total_rows * sample_rate)
        skip_rows = sorted(random.sample(range(1, total_rows + 1), total_rows - rows_to_read))
        
        df = pd.read_csv(uploaded_file, skiprows=skip_rows)
        st.success(f"✅ Loaded {sample_rate:.0%} sample of data ({len(df):,} rows)")
        st.info(f"💡 Original data has approximately {total_rows:,} rows")
        return df
        
    except Exception as e:
        st.error(f"❌ Failed to load sampled data: {str(e)}")
        return None
