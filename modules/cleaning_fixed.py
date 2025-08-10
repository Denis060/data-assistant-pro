# Enhanced modules/cleaning.py

import logging
import time

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from .cache_utils import with_progress_cache

# Setup logger
logger = logging.getLogger(__name__)


def detect_outliers_iqr(data, multiplier=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Args:
        data: Series or array of numerical data
        multiplier: The multiplier for IQR (default 1.5)

    Returns:
        Dictionary with outlier mask and detailed information
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outlier_mask = (data < lower_bound) | (data > upper_bound)

    # Get outlier details
    outlier_indices = data[outlier_mask].index.tolist()
    outlier_values = data[outlier_mask].values.tolist()

    return {
        "mask": outlier_mask,
        "indices": outlier_indices,
        "values": outlier_values,
        "method": "IQR",
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "multiplier": multiplier,
        "total_outliers": len(outlier_indices),
        "explanation": f"Values below {lower_bound:.2f} or above {upper_bound:.2f} are outliers (Q1-{multiplier}*IQR to Q3+{multiplier}*IQR)",
    }


def detect_outliers_zscore(data, threshold=3):
    """
    Detect outliers using the Z-score method.

    Args:
        data: Series or array of numerical data
        threshold: The Z-score threshold (default 3)

    Returns:
        Dictionary with outlier mask and detailed information
    """
    from scipy import stats

    z_scores = np.abs(stats.zscore(data.dropna()))
    mean_val = data.mean()
    std_val = data.std()

    # Create full mask including NaN positions
    outlier_mask = pd.Series(False, index=data.index)
    outlier_mask[data.dropna().index] = z_scores > threshold

    # Get outlier details
    outlier_indices = data[outlier_mask].index.tolist()
    outlier_values = data[outlier_mask].values.tolist()
    outlier_zscores = z_scores[z_scores > threshold].tolist()

    return {
        "mask": outlier_mask,
        "indices": outlier_indices,
        "values": outlier_values,
        "z_scores": outlier_zscores,
        "method": "Z-Score",
        "mean": mean_val,
        "std": std_val,
        "threshold": threshold,
        "total_outliers": len(outlier_indices),
        "explanation": f"Values with |Z-score| > {threshold} are outliers (mean={mean_val:.2f}, std={std_val:.2f})",
    }


def detect_outliers_modified_zscore(data, threshold=3.5):
    """
    Detect outliers using the Modified Z-score method (more robust).

    Args:
        data: Series or array of numerical data
        threshold: The modified Z-score threshold (default 3.5)

    Returns:
        Dictionary with outlier mask and detailed information
    """
    median_val = data.median()
    mad = np.median(np.abs(data - median_val))
    modified_z_scores = (
        0.6745 * (data - median_val) / mad if mad != 0 else np.zeros_like(data)
    )
    outlier_mask = np.abs(modified_z_scores) > threshold

    # Get outlier details
    outlier_indices = data[outlier_mask].index.tolist()
    outlier_values = data[outlier_mask].values.tolist()
    outlier_mod_zscores = modified_z_scores[outlier_mask].tolist()

    return {
        "mask": outlier_mask,
        "indices": outlier_indices,
        "values": outlier_values,
        "modified_z_scores": outlier_mod_zscores,
        "method": "Modified Z-Score",
        "median": median_val,
        "mad": mad,
        "threshold": threshold,
        "total_outliers": len(outlier_indices),
        "explanation": f"Values with |Modified Z-score| > {threshold} are outliers (median={median_val:.2f}, MAD={mad:.2f})",
    }


def show_outlier_analysis(df, col, detection_method, **kwargs):
    """
    Show detailed outlier analysis with visualizations and explanations.
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = df[col].copy()

    # Detect outliers using the specified method
    if detection_method == "IQR":
        multiplier = kwargs.get("iqr_multiplier", 1.5)
        outlier_info = detect_outliers_iqr(data, multiplier)
    elif detection_method == "Z-Score":
        threshold = kwargs.get("zscore_threshold", 3)
        outlier_info = detect_outliers_zscore(data, threshold)
    elif detection_method == "Modified Z-Score":
        threshold = kwargs.get("modified_zscore_threshold", 3.5)
        outlier_info = detect_outliers_modified_zscore(data, threshold)
    else:
        return None

    outlier_mask = outlier_info["mask"]

    # Show outlier analysis
    st.write(f"### 🔍 Outlier Analysis for '{col}'")
    st.write(f"**Method**: {outlier_info['method']}")
    st.write(f"**Explanation**: {outlier_info['explanation']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data Points", len(data))
    with col2:
        st.metric("Outliers Found", outlier_info["total_outliers"])
    with col3:
        outlier_percentage = (outlier_info["total_outliers"] / len(data)) * 100
        st.metric("Outlier Percentage", f"{outlier_percentage:.1f}%")

    # Show detailed outlier information
    if outlier_info["total_outliers"] > 0:
        st.write("### 📊 Outlier Details")

        # Create detailed outlier DataFrame
        outlier_details = pd.DataFrame(
            {"Index": outlier_info["indices"], "Value": outlier_info["values"]}
        )

        # Add method-specific details
        if detection_method == "Z-Score" and "z_scores" in outlier_info:
            outlier_details["Z-Score"] = outlier_info["z_scores"]
        elif (
            detection_method == "Modified Z-Score"
            and "modified_z_scores" in outlier_info
        ):
            outlier_details["Modified Z-Score"] = outlier_info["modified_z_scores"]
        elif detection_method == "IQR":
            # Add distance from bounds for IQR
            distances = []
            for val in outlier_info["values"]:
                if val < outlier_info["lower_bound"]:
                    distances.append(
                        f"Below by {outlier_info['lower_bound'] - val:.2f}"
                    )
                else:
                    distances.append(
                        f"Above by {val - outlier_info['upper_bound']:.2f}"
                    )
            outlier_details["Distance from Bound"] = distances

        # Show outlier table (limit to first 20 for display)
        if len(outlier_details) > 20:
            st.write(
                f"Showing first 20 outliers (out of {len(outlier_details)} total):"
            )
            st.dataframe(outlier_details.head(20))
        else:
            st.dataframe(outlier_details)

    # Create visualization
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Distribution Histogram",
            "Box Plot with Outliers",
            "Data Points (Outliers Highlighted)",
            "Statistical Summary",
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data,
            name="Data Distribution",
            nbinsx=30,
            marker_color="lightblue",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    # Box plot
    fig.add_trace(
        go.Box(y=data, name="Box Plot", marker_color="lightgreen"), row=1, col=2
    )

    # Scatter plot highlighting outliers
    normal_data = data[outlier_mask == False]
    outlier_data = data[outlier_mask]

    fig.add_trace(
        go.Scatter(
            x=normal_data.index,
            y=normal_data,
            mode="markers",
            name="Normal Data",
            marker=dict(color="blue", size=4, opacity=0.6),
        ),
        row=2,
        col=1,
    )

    if len(outlier_data) > 0:
        fig.add_trace(
            go.Scatter(
                x=outlier_data.index,
                y=outlier_data,
                mode="markers",
                name="Outliers",
                marker=dict(color="red", size=8, symbol="x"),
            ),
            row=2,
            col=1,
        )

    # Statistical summary table
    if detection_method == "IQR":
        summary_data = [
            ["Q1", f"{outlier_info['Q1']:.2f}"],
            ["Q3", f"{outlier_info['Q3']:.2f}"],
            ["IQR", f"{outlier_info['IQR']:.2f}"],
            ["Lower Bound", f"{outlier_info['lower_bound']:.2f}"],
            ["Upper Bound", f"{outlier_info['upper_bound']:.2f}"],
            ["Multiplier", f"{outlier_info['multiplier']}"],
        ]
    elif detection_method == "Z-Score":
        summary_data = [
            ["Mean", f"{outlier_info['mean']:.2f}"],
            ["Std Dev", f"{outlier_info['std']:.2f}"],
            ["Threshold", f"{outlier_info['threshold']}"],
            [
                "Upper Limit",
                f"{outlier_info['mean'] + outlier_info['threshold'] * outlier_info['std']:.2f}",
            ],
            [
                "Lower Limit",
                f"{outlier_info['mean'] - outlier_info['threshold'] * outlier_info['std']:.2f}",
            ],
        ]
    else:  # Modified Z-Score
        summary_data = [
            ["Median", f"{outlier_info['median']:.2f}"],
            ["MAD", f"{outlier_info['mad']:.2f}"],
            ["Threshold", f"{outlier_info['threshold']}"],
            ["Method", "Modified Z-Score (Robust)"],
        ]

    fig.add_trace(
        go.Table(
            header=dict(values=["Statistic", "Value"], fill_color="lightgray"),
            cells=dict(values=list(zip(*summary_data)), fill_color="white"),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        title_text=f"Comprehensive Outlier Analysis for '{col}'",
    )
    st.plotly_chart(fig, use_container_width=True)

    return outlier_info


def handle_outliers(
    df: pd.DataFrame, method: str, detection_method: str, columns: list, **kwargs
) -> pd.DataFrame:
    """
    Comprehensive outlier detection and treatment with detailed analysis.

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
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        df_cleaned = df.copy()

        if not columns:
            progress_bar.empty()
            status_text.empty()
            st.warning("Please select one or more columns for outlier detection.")
            return df_cleaned

        status_text.text("🔍 Initializing outlier detection...")
        progress_bar.progress(10)

        # Show detailed analysis for each column
        total_columns = len(columns)
        for i, col in enumerate(columns):
            # Update progress
            progress = 10 + (i / total_columns) * 70  # Reserve last 20% for final steps
            status_text.text(f"🔍 Processing column {i+1}/{total_columns}: {col}")
            progress_bar.progress(int(progress))
            
            if col not in df.columns:
                st.warning(f"Column '{col}' not found in data.")
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                st.warning(
                    f"Column '{col}' is not numeric. Skipping outlier detection."
                )
                continue

            # Show comprehensive outlier analysis
            outlier_info = show_outlier_analysis(df, col, detection_method, **kwargs)
            if outlier_info is None:
                continue

            outlier_mask = outlier_info["mask"]

            if outlier_info["total_outliers"] == 0:
                st.success(
                    f"✅ No outliers detected in '{col}' using {detection_method} method"
                )
                continue

            # Apply treatment method with detailed feedback
            st.write(f"### 🛠️ Applying Treatment: {method}")

            initial_stats = {
                "mean": df_cleaned[col].mean(),
                "median": df_cleaned[col].median(),
                "std": df_cleaned[col].std(),
                "min": df_cleaned[col].min(),
                "max": df_cleaned[col].max(),
                "count": len(df_cleaned),
            }

            if method == "Remove":
                df_cleaned = df_cleaned[outlier_mask == False]
                treatment_msg = f"Removed {outlier_info['total_outliers']} outlier rows"

            elif method == "Cap":
                if detection_method == "IQR":
                    lower_bound = outlier_info["lower_bound"]
                    upper_bound = outlier_info["upper_bound"]
                    df_cleaned.loc[
                        outlier_mask & (df_cleaned[col] < lower_bound), col
                    ] = lower_bound
                    df_cleaned.loc[
                        outlier_mask & (df_cleaned[col] > upper_bound), col
                    ] = upper_bound
                    treatment_msg = f"Capped {outlier_info['total_outliers']} outliers to bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
                else:
                    # For Z-score methods, cap to reasonable percentiles
                    lower_percentile = df[col].quantile(0.05)
                    upper_percentile = df[col].quantile(0.95)
                    df_cleaned.loc[
                        outlier_mask & (df_cleaned[col] < lower_percentile), col
                    ] = lower_percentile
                    df_cleaned.loc[
                        outlier_mask & (df_cleaned[col] > upper_percentile), col
                    ] = upper_percentile
                    treatment_msg = f"Capped {outlier_info['total_outliers']} outliers to 5th-95th percentile range"

            elif method == "Transform":
                if (df_cleaned[col] > 0).all():
                    df_cleaned[col] = np.log1p(df_cleaned[col])
                    treatment_msg = f"Applied log transformation to '{col}' (including {outlier_info['total_outliers']} outliers)"
                else:
                    st.warning(
                        f"Cannot apply log transformation to '{col}' due to non-positive values."
                    )
                    continue

            elif method == "Replace with Mean":
                mean_val = df[col][outlier_mask == False].mean()  # Mean of non-outliers
                df_cleaned.loc[outlier_mask, col] = mean_val
                treatment_msg = f"Replaced {outlier_info['total_outliers']} outliers with non-outlier mean ({mean_val:.2f})"

            elif method == "Replace with Median":
                median_val = df[col][outlier_mask == False].median()  # Median of non-outliers
                df_cleaned.loc[outlier_mask, col] = median_val
                treatment_msg = f"Replaced {outlier_info['total_outliers']} outliers with non-outlier median ({median_val:.2f})"

            st.success(f"✅ {treatment_msg}")

            # Show before/after comparison
            final_stats = {
                "mean": df_cleaned[col].mean(),
                "median": df_cleaned[col].median(),
                "std": df_cleaned[col].std(),
                "min": df_cleaned[col].min(),
                "max": df_cleaned[col].max(),
                "count": len(df_cleaned),
            }

            st.write("### 📊 Impact Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Before Treatment:**")
                st.write(f"• Count: {initial_stats['count']:,}")
                st.write(f"• Mean: {initial_stats['mean']:.2f}")
                st.write(f"• Median: {initial_stats['median']:.2f}")
                st.write(f"• Std Dev: {initial_stats['std']:.2f}")
                st.write(
                    f"• Range: [{initial_stats['min']:.2f}, {initial_stats['max']:.2f}]"
                )

            with col2:
                st.write("**After Treatment:**")
                st.write(f"• Count: {final_stats['count']:,}")
                st.write(f"• Mean: {final_stats['mean']:.2f}")
                st.write(f"• Median: {final_stats['median']:.2f}")
                st.write(f"• Std Dev: {final_stats['std']:.2f}")
                st.write(
                    f"• Range: [{final_stats['min']:.2f}, {final_stats['max']:.2f}]"
                )

            # Show changes
            st.write("**Changes:**")
            mean_change = (
                (final_stats["mean"] - initial_stats["mean"]) / initial_stats["mean"]
            ) * 100
            std_change = (
                (final_stats["std"] - initial_stats["std"]) / initial_stats["std"]
            ) * 100
            st.write(f"• Mean change: {mean_change:+.1f}%")
            st.write(f"• Std Dev change: {std_change:+.1f}%")
            if method == "Remove":
                st.write(
                    f"• Rows removed: {initial_stats['count'] - final_stats['count']:,}"
                )

            st.divider()

        # Final progress update
        status_text.text("✅ Outlier treatment completed!")
        progress_bar.progress(100)
        time.sleep(0.3)  # Brief pause for visual feedback
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()

        logger.info(
            f"Outlier treatment completed using {detection_method} detection and {method} treatment"
        )
        return df_cleaned

    except Exception as e:
        st.error(f"Error in outlier handling: {str(e)}")
        logger.error(f"Outlier handling error: {str(e)}")
        return df


def handle_missing_values(
    df: pd.DataFrame, strategy: str, columns: list
) -> pd.DataFrame:
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
            st.warning(
                "Please select one or more columns to apply the cleaning strategy."
            )
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
                df_cleaned[col] = df_cleaned[col].ffill()

        elif strategy == "Backward Fill":
            for col in columns:
                df_cleaned[col] = df_cleaned[col].bfill()

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
                        st.warning(
                            f"Cannot calculate mean for non-numeric column '{col}'. Skipping."
                        )

                elif strategy == "Fill with Median":
                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        median_value = df_cleaned[col].median()
                        df_cleaned[col].fillna(median_value, inplace=True)
                        logger.info(f"Filled {col} with median: {median_value:.2f}")
                    else:
                        st.warning(
                            f"Cannot calculate median for non-numeric column '{col}'. Skipping."
                        )

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
        return df.to_csv(index=False).encode("utf-8")
    except Exception as e:
        logger.error(f"Error converting DataFrame to CSV: {str(e)}")
        st.error(f"Error converting to CSV: {str(e)}")
        return ""
