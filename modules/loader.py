"""Enhanced data loading utilities for Data Assistant Pro."""

import io
import logging
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

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
            # Log file upload
            logger.info(
                f"Loading file: {uploaded_file.name}, Size: {uploaded_file.size} bytes"
            )

            # File size validation (max 100MB)
            if uploaded_file.size > 100 * 1024 * 1024:
                st.error("❌ File too large! Please upload a file smaller than 100MB.")
                return None

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
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=delimiter)
                delimiter_source = "manually selected"
            else:
                # Use smart CSV reading with delimiter detection
                df, delimiter = smart_read_csv(uploaded_file)
                delimiter_source = "auto-detected"

            if df is None:
                return None

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
        except pd.errors.ParserError as e:
            st.error(f"❌ Error parsing CSV file: {str(e)}")
            logger.error(f"Parser error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error loading file: {str(e)}")
            logger.error(f"Unexpected error loading file: {str(e)}")
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
