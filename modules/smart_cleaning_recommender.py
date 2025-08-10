"""
Smart Cleaning Strategy Recommender
Analyzes data characteristics and recommends optimal cleaning approaches
Prioritizes strategies that will most improve ML model performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class SmartCleaningRecommender:
    """
    Intelligent system that analyzes data and recommends optimal cleaning strategies.
    Focuses on improvements that will have the biggest impact on ML model performance.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str = None):
        self.df = df
        self.target_column = target_column
        self.recommendations = []
        
    def analyze_and_recommend(self) -> Dict:
        """
        Analyze data characteristics and recommend prioritized cleaning strategies.
        
        Returns:
            Dict with recommendations, priorities, and expected improvements
        """
        
        # Analyze different aspects of data quality
        missing_analysis = self._analyze_missing_patterns()
        outlier_analysis = self._analyze_outlier_impact()
        encoding_analysis = self._analyze_encoding_needs()
        feature_analysis = self._analyze_feature_quality()
        ml_readiness = self._analyze_ml_readiness()
        
        # Generate prioritized recommendations
        all_recommendations = []
        all_recommendations.extend(missing_analysis['recommendations'])
        all_recommendations.extend(outlier_analysis['recommendations'])
        all_recommendations.extend(encoding_analysis['recommendations'])
        all_recommendations.extend(feature_analysis['recommendations'])
        all_recommendations.extend(ml_readiness['recommendations'])
        
        # Sort by priority and expected impact
        prioritized_recommendations = sorted(
            all_recommendations, 
            key=lambda x: (x['priority'], -x['expected_improvement']), 
            reverse=True
        )
        
        return {
            'recommendations': prioritized_recommendations,
            'analysis_summary': {
                'missing_data': missing_analysis['summary'],
                'outliers': outlier_analysis['summary'],
                'encoding': encoding_analysis['summary'],
                'features': feature_analysis['summary'],
                'ml_readiness': ml_readiness['summary']
            },
            'overall_strategy': self._determine_overall_strategy(prioritized_recommendations),
            'expected_improvement': sum([r['expected_improvement'] for r in prioritized_recommendations[:5]])  # Top 5
        }
    
    def _analyze_missing_patterns(self) -> Dict:
        """Analyze missing data patterns and recommend strategies."""
        
        missing_analysis = {
            'recommendations': [],
            'summary': {}
        }
        
        total_missing = self.df.isnull().sum().sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_percentage = (total_missing / total_cells) * 100
        
        missing_analysis['summary'] = {
            'total_missing': total_missing,
            'missing_percentage': missing_percentage,
            'columns_with_missing': (self.df.isnull().sum() > 0).sum()
        }
        
        if missing_percentage > 20:
            # High missing data - recommend aggressive strategy
            missing_analysis['recommendations'].append({
                'category': 'Missing Data',
                'action': 'Use ML-Ready cleaning with advanced imputation',
                'reason': f'{missing_percentage:.1f}% missing data requires sophisticated handling',
                'priority': 5,  # High priority
                'expected_improvement': 25,
                'implementation': 'Use KNN imputation for numeric, create missing indicators for high-missingness columns'
            })
        elif missing_percentage > 5:
            # Medium missing data
            missing_analysis['recommendations'].append({
                'category': 'Missing Data',
                'action': 'Use Enhanced Conservative cleaning',
                'reason': f'{missing_percentage:.1f}% missing data can be handled with standard methods',
                'priority': 3,  # Medium priority
                'expected_improvement': 15,
                'implementation': 'Use mean/median for numeric, mode for categorical'
            })
        
        # Check for columns with excessive missing data
        high_missing_cols = self.df.columns[self.df.isnull().sum() / len(self.df) > 0.5]
        if len(high_missing_cols) > 0:
            missing_analysis['recommendations'].append({
                'category': 'Data Quality',
                'action': f'Consider dropping columns: {list(high_missing_cols)}',
                'reason': 'Columns with >50% missing data rarely improve model performance',
                'priority': 4,  # High priority
                'expected_improvement': 20,
                'implementation': 'Drop columns before training, or use as separate indicator variables'
            })
        
        return missing_analysis
    
    def _analyze_outlier_impact(self) -> Dict:
        """Analyze outlier patterns and recommend treatment strategies."""
        
        outlier_analysis = {
            'recommendations': [],
            'summary': {}
        }
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        outlier_cols = []
        
        for col in numeric_cols:
            if col == self.target_column:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | 
                           (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_percentage = outliers / len(self.df) * 100
                
                if outlier_percentage > 5:
                    total_outliers += outliers
                    outlier_cols.append(col)
        
        outlier_analysis['summary'] = {
            'total_outliers': total_outliers,
            'affected_columns': len(outlier_cols),
            'outlier_percentage': total_outliers / (len(self.df) * len(numeric_cols)) * 100 if len(numeric_cols) > 0 else 0
        }
        
        if len(outlier_cols) > 0:
            if total_outliers > len(self.df) * 0.1:  # More than 10% outliers
                outlier_analysis['recommendations'].append({
                    'category': 'Outliers',
                    'action': 'Use ML-Ready cleaning with advanced outlier handling',
                    'reason': f'High outlier count in {len(outlier_cols)} columns may skew model performance',
                    'priority': 4,  # High priority
                    'expected_improvement': 15,
                    'implementation': 'Use capping, transformation, and outlier indicators'
                })
            else:
                outlier_analysis['recommendations'].append({
                    'category': 'Outliers',
                    'action': 'Use Enhanced Aggressive cleaning',
                    'reason': f'Moderate outliers detected in {len(outlier_cols)} columns',
                    'priority': 3,  # Medium priority
                    'expected_improvement': 10,
                    'implementation': 'Cap outliers using IQR method'
                })
        
        return outlier_analysis
    
    def _analyze_encoding_needs(self) -> Dict:
        """Analyze categorical data and recommend encoding strategies."""
        
        encoding_analysis = {
            'recommendations': [],
            'summary': {}
        }
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        high_cardinality_cols = []
        needs_encoding = []
        
        for col in categorical_cols:
            if col == self.target_column:
                continue
                
            unique_count = self.df[col].nunique()
            cardinality_ratio = unique_count / len(self.df)
            
            if unique_count > 50:
                high_cardinality_cols.append(col)
            if unique_count > 2:  # More than binary
                needs_encoding.append(col)
        
        encoding_analysis['summary'] = {
            'categorical_columns': len(categorical_cols),
            'high_cardinality': len(high_cardinality_cols),
            'needs_encoding': len(needs_encoding)
        }
        
        if len(high_cardinality_cols) > 0:
            encoding_analysis['recommendations'].append({
                'category': 'Feature Engineering',
                'action': 'Use ML-Ready cleaning for high-cardinality features',
                'reason': f'{len(high_cardinality_cols)} columns have high cardinality (>50 unique values)',
                'priority': 4,  # High priority
                'expected_improvement': 20,
                'implementation': 'Group rare categories, consider target encoding'
            })
        
        if len(needs_encoding) > 0:
            encoding_analysis['recommendations'].append({
                'category': 'Data Preparation',
                'action': 'Ensure categorical encoding before modeling',
                'reason': f'{len(needs_encoding)} categorical columns need encoding for ML',
                'priority': 5,  # Critical priority
                'expected_improvement': 30,
                'implementation': 'Use label encoding or one-hot encoding'
            })
        
        return encoding_analysis
    
    def _analyze_feature_quality(self) -> Dict:
        """Analyze feature quality and recommend improvements."""
        
        feature_analysis = {
            'recommendations': [],
            'summary': {}
        }
        
        # Check for constant features
        constant_features = []
        nearly_constant_features = []
        
        for col in self.df.columns:
            if col == self.target_column:
                continue
                
            unique_count = self.df[col].nunique()
            if unique_count <= 1:
                constant_features.append(col)
            elif unique_count > 1:
                # Check if nearly constant (>95% same value)
                mode_frequency = self.df[col].value_counts().iloc[0] / len(self.df)
                if mode_frequency > 0.95:
                    nearly_constant_features.append(col)
        
        feature_analysis['summary'] = {
            'constant_features': len(constant_features),
            'nearly_constant_features': len(nearly_constant_features),
            'total_features': len(self.df.columns)
        }
        
        if len(constant_features) > 0:
            feature_analysis['recommendations'].append({
                'category': 'Feature Selection',
                'action': f'Remove constant features: {constant_features}',
                'reason': 'Constant features provide no information for ML models',
                'priority': 5,  # Critical priority
                'expected_improvement': 15,
                'implementation': 'Drop these columns before training'
            })
        
        if len(nearly_constant_features) > 0:
            feature_analysis['recommendations'].append({
                'category': 'Feature Selection',
                'action': f'Consider removing nearly constant features: {nearly_constant_features}',
                'reason': 'Nearly constant features provide minimal information',
                'priority': 2,  # Low priority
                'expected_improvement': 5,
                'implementation': 'Evaluate if these features add value to your specific use case'
            })
        
        return feature_analysis
    
    def _analyze_ml_readiness(self) -> Dict:
        """Analyze overall ML readiness and recommend final preparations."""
        
        ml_analysis = {
            'recommendations': [],
            'summary': {}
        }
        
        # Check data size
        sample_size = len(self.df)
        feature_count = len(self.df.columns)
        
        # Check data types
        numeric_ratio = len(self.df.select_dtypes(include=[np.number]).columns) / feature_count
        
        ml_analysis['summary'] = {
            'sample_size': sample_size,
            'feature_count': feature_count,
            'numeric_ratio': numeric_ratio
        }
        
        if sample_size < 1000:
            ml_analysis['recommendations'].append({
                'category': 'Data Collection',
                'action': 'Consider collecting more data',
                'reason': f'Sample size ({sample_size}) may be too small for robust ML models',
                'priority': 2,  # Low priority (can't always fix)
                'expected_improvement': 25,
                'implementation': 'Aim for at least 1000+ samples, or use simpler models'
            })
        
        if numeric_ratio < 0.5:  # Less than 50% numeric
            ml_analysis['recommendations'].append({
                'category': 'Feature Engineering',
                'action': 'Use ML-Ready cleaning for comprehensive feature engineering',
                'reason': f'Only {numeric_ratio:.1%} of features are numeric - need better encoding',
                'priority': 4,  # High priority
                'expected_improvement': 20,
                'implementation': 'Convert categorical to numeric, create derived features'
            })
        
        return ml_analysis
    
    def _determine_overall_strategy(self, recommendations: List[Dict]) -> str:
        """Determine the overall recommended cleaning strategy."""
        
        if not recommendations:
            return "Conservative"
        
        # Count high-priority recommendations
        high_priority_count = sum(1 for r in recommendations if r['priority'] >= 4)
        total_expected_improvement = sum(r['expected_improvement'] for r in recommendations[:3])
        
        if high_priority_count >= 3 or total_expected_improvement > 50:
            return "ML-Ready"
        elif high_priority_count >= 1 or total_expected_improvement > 25:
            return "Aggressive"
        else:
            return "Conservative"

def generate_cleaning_strategy_report(df: pd.DataFrame, target_column: str = None) -> Dict:
    """
    Generate a comprehensive cleaning strategy report with prioritized recommendations.
    
    Args:
        df: DataFrame to analyze
        target_column: Target variable for ML analysis
        
    Returns:
        Dict with detailed recommendations and strategy
    """
    
    recommender = SmartCleaningRecommender(df, target_column)
    analysis = recommender.analyze_and_recommend()
    
    return analysis
