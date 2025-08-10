"""
Enhanced AI-Powered Data Insights with Advanced Intelligence
Building on your existing excellent foundation with next-level AI capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedAIEngine:
    """
    Next-generation AI engine for enhanced data intelligence
    Building on your existing excellent data quality and insights modules
    """
    
    def __init__(self):
        self.ai_models = {}
        self.insight_cache = {}
        self.advanced_patterns = {}
    
    def enhanced_anomaly_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Multi-algorithm anomaly detection with confidence scoring
        """
        anomaly_results = {
            'isolation_forest': {},
            'statistical': {},
            'clustering_based': {},
            'ensemble_score': {},
            'actionable_insights': []
        }
        
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) == 0:
            return {'message': 'No numeric columns for anomaly detection'}
        
        # Fill missing values for analysis
        numeric_df_filled = numeric_df.fillna(numeric_df.median())
        
        # 1. Isolation Forest (Enhanced)
        iso_forest = IsolationForest(
            contamination=0.1, 
            random_state=42, 
            n_estimators=200
        )
        isolation_scores = iso_forest.fit_predict(numeric_df_filled)
        isolation_confidence = iso_forest.decision_function(numeric_df_filled)
        
        # 2. Statistical Anomaly Detection (Z-score + Modified Z-score)
        z_scores = np.abs(stats.zscore(numeric_df_filled, axis=0, nan_policy='omit'))
        modified_z_scores = self._calculate_modified_zscore(numeric_df_filled)
        
        # 3. Clustering-based Anomaly Detection
        if len(numeric_df_filled) > 10:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df_filled)
            
            n_clusters = min(8, len(numeric_df_filled) // 5)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_data)
                
                # Calculate distance to cluster centers
                distances = []
                for i, point in enumerate(scaled_data):
                    center = kmeans.cluster_centers_[cluster_labels[i]]
                    distance = np.linalg.norm(point - center)
                    distances.append(distance)
                
                # Anomalies are points far from their cluster center
                distance_threshold = np.percentile(distances, 90)
                clustering_anomalies = np.array(distances) > distance_threshold
        
        # Ensemble scoring
        ensemble_scores = []
        for i in range(len(df)):
            score = 0
            
            # Isolation forest contribution
            if isolation_scores[i] == -1:
                score += 0.4
            
            # Statistical contribution
            if (z_scores[i] > 3).any() or (modified_z_scores[i] > 3.5).any():
                score += 0.3
            
            # Clustering contribution
            if len(numeric_df_filled) > 10 and n_clusters >= 2:
                if clustering_anomalies[i]:
                    score += 0.3
            
            ensemble_scores.append(score)
        
        # Generate actionable insights
        high_confidence_anomalies = [i for i, score in enumerate(ensemble_scores) if score > 0.6]
        
        anomaly_results['actionable_insights'] = [
            f"🚨 {len(high_confidence_anomalies)} high-confidence anomalies detected",
            f"📊 Isolation Forest flagged {np.sum(isolation_scores == -1)} records",
            f"📈 Statistical methods flagged {np.sum((z_scores > 3).any(axis=1))} records",
            "🔍 Review flagged records for data quality issues or interesting patterns"
        ]
        
        anomaly_results['summary'] = {
            'total_anomalies': len(high_confidence_anomalies),
            'anomaly_percentage': (len(high_confidence_anomalies) / len(df)) * 100,
            'confidence_distribution': {
                'high': len([s for s in ensemble_scores if s > 0.6]),
                'medium': len([s for s in ensemble_scores if 0.3 <= s <= 0.6]),
                'low': len([s for s in ensemble_scores if s < 0.3])
            }
        }
        
        return anomaly_results
    
    def intelligent_feature_engineering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        AI-powered feature engineering suggestions
        """
        suggestions = {
            'new_features': [],
            'transformations': [],
            'interactions': [],
            'temporal_features': [],
            'statistical_features': []
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # 1. Ratio and Interaction Features
        if len(numeric_cols) >= 2:
            # Find meaningful ratios
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    if df[col1].std() > 0 and df[col2].std() > 0:
                        correlation = df[col1].corr(df[col2])
                        if abs(correlation) > 0.3:
                            suggestions['interactions'].append({
                                'type': 'ratio',
                                'formula': f"{col1} / {col2}",
                                'rationale': f"Ratio might capture relationship (corr: {correlation:.3f})"
                            })
        
        # 2. Polynomial Features for Non-linear Patterns
        for col in numeric_cols:
            if df[col].var() > 0:
                # Check for non-linear patterns
                linear_trend = np.corrcoef(range(len(df)), df[col].fillna(df[col].mean()))[0,1]
                if abs(linear_trend) < 0.3:  # Weak linear relationship
                    suggestions['transformations'].append({
                        'type': 'polynomial',
                        'feature': col,
                        'suggestion': f"{col}^2 or sqrt({col})",
                        'rationale': "Non-linear transformation may capture hidden patterns"
                    })
        
        # 3. Binning for Continuous Variables
        for col in numeric_cols:
            unique_vals = df[col].nunique()
            if unique_vals > 20:  # High cardinality
                suggestions['transformations'].append({
                    'type': 'binning',
                    'feature': col,
                    'suggestion': f"Create bins for {col}",
                    'rationale': f"High cardinality ({unique_vals} unique values) - binning may improve model performance"
                })
        
        # 4. One-hot Encoding for Categorical Variables
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if 2 <= unique_vals <= 10:  # Optimal range for one-hot encoding
                suggestions['new_features'].append({
                    'type': 'one_hot',
                    'feature': col,
                    'suggestion': f"One-hot encode {col}",
                    'rationale': f"Categorical with {unique_vals} categories - good for encoding"
                })
        
        # 5. Temporal Features (if date columns exist)
        date_cols = []
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    pass
        
        for col in date_cols:
            suggestions['temporal_features'].extend([
                f"Extract day_of_week from {col}",
                f"Extract month from {col}",
                f"Extract quarter from {col}",
                f"Calculate days_since from {col}"
            ])
        
        return suggestions
    
    def predictive_data_profiling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        AI-powered predictive profiling to suggest next steps
        """
        profile = {
            'dataset_personality': {},
            'ml_readiness': {},
            'recommended_algorithms': [],
            'data_story': ""
        }
        
        # Dataset personality assessment
        n_rows, n_cols = df.shape
        numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / n_cols
        missing_ratio = df.isnull().sum().sum() / (n_rows * n_cols)
        
        # Classify dataset personality
        if n_rows > 100000:
            size_personality = "Big Data"
        elif n_rows > 10000:
            size_personality = "Large Dataset"
        elif n_rows > 1000:
            size_personality = "Medium Dataset"
        else:
            size_personality = "Small Dataset"
        
        if missing_ratio < 0.05:
            quality_personality = "Clean & Complete"
        elif missing_ratio < 0.15:
            quality_personality = "Good Quality"
        else:
            quality_personality = "Needs Cleaning"
        
        if numeric_ratio > 0.8:
            type_personality = "Highly Numerical"
        elif numeric_ratio > 0.5:
            type_personality = "Mixed Types"
        else:
            type_personality = "Categorical Heavy"
        
        profile['dataset_personality'] = {
            'size': size_personality,
            'quality': quality_personality,
            'type': type_personality,
            'complexity_score': self._calculate_complexity_score(df)
        }
        
        # ML readiness assessment
        readiness_score = 0
        readiness_factors = []
        
        if missing_ratio < 0.1:
            readiness_score += 25
            readiness_factors.append("✅ Low missing data")
        else:
            readiness_factors.append("⚠️ High missing data - preprocessing needed")
        
        if n_rows > 1000:
            readiness_score += 25
            readiness_factors.append("✅ Sufficient sample size")
        else:
            readiness_factors.append("⚠️ Small sample size - use cross-validation")
        
        if len(df.select_dtypes(include=[np.number]).columns) >= 3:
            readiness_score += 25
            readiness_factors.append("✅ Multiple numeric features")
        
        # Check for class balance (if target column is obvious)
        potential_targets = [col for col in df.columns if any(term in col.lower() 
                            for term in ['target', 'label', 'class', 'outcome', 'y'])]
        if potential_targets:
            target_col = potential_targets[0]
            if df[target_col].dtype == 'object':
                value_counts = df[target_col].value_counts()
                balance_ratio = value_counts.min() / value_counts.max()
                if balance_ratio > 0.3:
                    readiness_score += 25
                    readiness_factors.append("✅ Reasonably balanced target")
                else:
                    readiness_factors.append("⚠️ Imbalanced target - consider resampling")
        
        profile['ml_readiness'] = {
            'score': readiness_score,
            'grade': self._get_readiness_grade(readiness_score),
            'factors': readiness_factors
        }
        
        # Algorithm recommendations based on data characteristics
        if size_personality == "Big Data":
            profile['recommended_algorithms'].extend([
                "Random Forest (handles large data well)",
                "Gradient Boosting (XGBoost, LightGBM)",
                "Neural Networks (if sufficient complexity)"
            ])
        elif size_personality == "Small Dataset":
            profile['recommended_algorithms'].extend([
                "Logistic Regression (simple, interpretable)",
                "Support Vector Machine (works well with small data)",
                "k-Nearest Neighbors (non-parametric)"
            ])
        else:
            profile['recommended_algorithms'].extend([
                "Random Forest (versatile choice)",
                "Gradient Boosting (often performs well)",
                "Ensemble Methods (combine multiple models)"
            ])
        
        # Generate AI narrative
        profile['data_story'] = f"""
        📊 **Your Dataset Story:**
        
        This is a {size_personality.lower()} with {quality_personality.lower()} characteristics and {type_personality.lower()} features. 
        
        **AI Assessment:** {profile['ml_readiness']['grade']} readiness for machine learning.
        
        **Key Insights:**
        - Dataset personality: {size_personality} × {quality_personality} × {type_personality}
        - Complexity score: {profile['dataset_personality']['complexity_score']:.2f}/1.0
        - ML readiness: {readiness_score}% ({profile['ml_readiness']['grade']})
        
        **Recommended Next Steps:**
        1. {profile['recommended_algorithms'][0] if profile['recommended_algorithms'] else 'Data preprocessing'}
        2. Feature engineering based on AI suggestions
        3. Cross-validation for robust evaluation
        """
        
        return profile
    
    def _calculate_modified_zscore(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate modified Z-score using median absolute deviation"""
        median = df.median()
        mad = ((df - median).abs()).median()
        modified_z_scores = 0.6745 * (df - median) / mad
        return np.abs(modified_z_scores)
    
    def _calculate_complexity_score(self, df: pd.DataFrame) -> float:
        """Calculate dataset complexity score"""
        n_rows, n_cols = df.shape
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        
        # Normalized metrics
        size_complexity = min(1.0, (n_rows * n_cols) / 1000000)  # Normalize by 1M cells
        type_diversity = min(1.0, len(set(df.dtypes)) / 5)  # Normalize by 5 types
        unique_ratio = df.nunique().mean() / len(df)
        
        # Correlation complexity for numeric data
        corr_complexity = 0
        if numeric_cols > 1:
            corr_matrix = df.select_dtypes(include=[np.number]).corr()
            corr_complexity = np.abs(corr_matrix).mean().mean()
        
        complexity = (
            size_complexity * 0.3 +
            type_diversity * 0.2 +
            unique_ratio * 0.3 +
            corr_complexity * 0.2
        )
        
        return min(1.0, complexity)
    
    def _get_readiness_grade(self, score: int) -> str:
        """Get ML readiness grade"""
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 80:
            return "A (Very Good)"
        elif score >= 70:
            return "B+ (Good)"
        elif score >= 60:
            return "B (Fair)"
        else:
            return "C (Needs Work)"

class SmartVisualizationEngine:
    """
    AI-powered visualization recommendations
    """
    
    @staticmethod
    def recommend_visualizations(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Intelligently recommend the best visualizations for the data
        """
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # 1. Distribution visualizations
        for col in numeric_cols:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                recommendations.append({
                    'type': 'histogram',
                    'columns': [col],
                    'rationale': f'{col} is skewed (skew={skewness:.2f}) - histogram shows distribution shape',
                    'priority': 'high'
                })
        
        # 2. Relationship visualizations
        if len(numeric_cols) >= 2:
            # Find highest correlation pairs
            corr_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.5:
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_val
                        ))
            
            for col1, col2, corr_val in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:3]:
                recommendations.append({
                    'type': 'scatter',
                    'columns': [col1, col2],
                    'rationale': f'Strong correlation ({corr_val:.3f}) between {col1} and {col2}',
                    'priority': 'high'
                })
        
        # 3. Categorical analysis
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 20:
                recommendations.append({
                    'type': 'bar_chart',
                    'columns': [col],
                    'rationale': f'{col} has {unique_count} categories - good for bar chart',
                    'priority': 'medium'
                })
        
        return recommendations
