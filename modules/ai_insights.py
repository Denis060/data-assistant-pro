"""
Advanced AI-Powered Insights Generator
Provides intelligent data interpretation and automated insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AIInsightsGenerator:
    """
    AI-powered insights generation for automated data analysis
    """
    
    def __init__(self):
        self.insights_cache = {}
        self.insight_templates = self._load_insight_templates()
    
    def generate_comprehensive_insights(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive AI-powered insights from data
        
        Args:
            df: Input DataFrame
            target_column: Optional target column for supervised insights
            
        Returns:
            Dictionary containing various insights and recommendations
        """
        try:
            insights = {
                'data_overview': self._generate_data_overview(df),
                'statistical_insights': self._generate_statistical_insights(df),
                'quality_insights': self._generate_quality_insights(df),
                'correlation_insights': self._generate_correlation_insights(df),
                'trend_insights': self._generate_trend_insights(df),
                'business_insights': self._generate_business_insights(df, target_column),
                'recommendations': self._generate_recommendations(df),
                'ai_narratives': self._generate_ai_narratives(df),
                'timestamp': datetime.now().isoformat()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {'error': str(e)}
    
    def _generate_data_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate intelligent data overview"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        overview = {
            'dataset_size': f"{len(df):,} rows × {len(df.columns)} columns",
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'missing_data_percentage': f"{(df.isnull().sum().sum() / df.size) * 100:.1f}%",
            'data_density': f"{((df.size - df.isnull().sum().sum()) / df.size) * 100:.1f}%"
        }
        
        # Add intelligent observations
        observations = []
        if len(df) > 100000:
            observations.append("Large dataset - consider sampling for initial exploration")
        if (df.isnull().sum().sum() / df.size) > 0.3:
            observations.append("High missing data percentage - data quality attention needed")
        if len(categorical_cols) > len(numeric_cols):
            observations.append("Categorical-heavy dataset - consider encoding strategies")
        
        overview['observations'] = observations
        return overview
    
    def _generate_statistical_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical insights with AI interpretation"""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {'message': 'No numeric columns for statistical analysis'}
        
        insights = {}
        
        # Distribution analysis
        for col in numeric_df.columns:
            data = numeric_df[col].dropna()
            if len(data) > 0:
                skewness = data.skew()
                kurtosis = data.kurtosis()
                
                # Intelligent interpretation
                distribution_type = "normal"
                if abs(skewness) > 1:
                    distribution_type = "highly skewed"
                elif abs(skewness) > 0.5:
                    distribution_type = "moderately skewed"
                
                insights[col] = {
                    'distribution_type': distribution_type,
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'outlier_percentage': self._calculate_outlier_percentage(data),
                    'interpretation': self._interpret_distribution(skewness, kurtosis)
                }
        
        return insights
    
    def _generate_quality_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality insights"""
        quality_issues = []
        
        # Check for various quality issues
        for col in df.columns:
            col_data = df[col]
            
            # Missing values
            missing_pct = (col_data.isnull().sum() / len(col_data)) * 100
            if missing_pct > 10:
                quality_issues.append(f"{col}: {missing_pct:.1f}% missing values")
            
            # Duplicates
            if col_data.dtype == 'object':
                unique_ratio = col_data.nunique() / len(col_data)
                if unique_ratio < 0.1 and col_data.nunique() < 10:
                    quality_issues.append(f"{col}: Low diversity ({unique_ratio:.1%} unique values)")
            
            # Constant values
            if col_data.nunique() == 1:
                quality_issues.append(f"{col}: Constant value (no variation)")
        
        # Overall quality score
        quality_score = max(0, 100 - len(quality_issues) * 5)
        
        return {
            'quality_score': quality_score,
            'quality_grade': self._get_quality_grade(quality_score),
            'issues': quality_issues,
            'recommendations': self._get_quality_recommendations(quality_issues)
        }
    
    def _generate_correlation_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate correlation insights"""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return {'message': 'Insufficient numeric columns for correlation analysis'}
        
        corr_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': 'strong' if abs(corr_val) > 0.8 else 'moderate'
                    })
        
        return {
            'strong_correlations': strong_correlations,
            'multicollinearity_risk': len([c for c in strong_correlations if abs(c['correlation']) > 0.9]),
            'insights': self._interpret_correlations(strong_correlations)
        }
    
    def _generate_trend_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trend insights for time-based data"""
        date_columns = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                date_columns.append(col)
        
        if not date_columns:
            return {'message': 'No time-based columns detected'}
        
        trends = {}
        for date_col in date_columns:
            try:
                df_sorted = df.sort_values(date_col)
                numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns
                
                for num_col in numeric_cols:
                    trend_direction = self._calculate_trend_direction(df_sorted[num_col])
                    trends[f"{num_col}_over_{date_col}"] = {
                        'direction': trend_direction,
                        'strength': self._calculate_trend_strength(df_sorted[num_col])
                    }
            except:
                continue
        
        return trends
    
    def _generate_business_insights(self, df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Generate business-focused insights"""
        business_insights = []
        
        # Revenue/Sales patterns
        revenue_cols = [col for col in df.columns if any(term in col.lower() 
                       for term in ['revenue', 'sales', 'income', 'profit', 'price'])]
        
        if revenue_cols:
            for col in revenue_cols:
                if df[col].dtype in [np.number]:
                    total_value = df[col].sum()
                    business_insights.append(f"Total {col}: ${total_value:,.2f}")
        
        # Customer patterns
        customer_cols = [col for col in df.columns if any(term in col.lower() 
                        for term in ['customer', 'client', 'user'])]
        
        if customer_cols:
            for col in customer_cols:
                unique_customers = df[col].nunique()
                business_insights.append(f"Unique {col}: {unique_customers:,}")
        
        # Growth insights
        if target_column and target_column in df.columns:
            target_analysis = self._analyze_target_variable(df, target_column)
            business_insights.extend(target_analysis)
        
        return {'insights': business_insights}
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        if missing_pct > 10:
            recommendations.append("Consider implementing data imputation strategies")
        
        # Feature engineering recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 10:
            recommendations.append("Consider dimensionality reduction (PCA, feature selection)")
        
        # Model recommendations
        if len(df) > 100000:
            recommendations.append("Large dataset - consider ensemble methods or deep learning")
        elif len(df) < 1000:
            recommendations.append("Small dataset - use cross-validation and simple models")
        
        return recommendations
    
    def _generate_ai_narratives(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate AI-powered narratives"""
        narratives = {}
        
        # Dataset story
        narratives['dataset_story'] = f"""
        This dataset contains {len(df):,} observations across {len(df.columns)} variables. 
        The data shows a {('well-structured' if df.isnull().sum().sum()/df.size < 0.1 else 'moderately structured')} 
        format with {df.select_dtypes(include=[np.number]).shape[1]} numeric and 
        {df.select_dtypes(include=['object']).shape[1]} categorical features.
        """
        
        # Quality narrative
        quality_score = max(0, 100 - (df.isnull().sum().sum() / df.size) * 100)
        narratives['quality_narrative'] = f"""
        The dataset demonstrates {('excellent' if quality_score > 90 else 'good' if quality_score > 70 else 'moderate')} 
        data quality with a {quality_score:.1f}% completeness score. 
        {'No significant data quality issues detected.' if quality_score > 90 else 'Some data quality attention recommended.'}
        """
        
        return narratives
    
    def _load_insight_templates(self) -> Dict[str, str]:
        """Load insight templates for AI generation"""
        return {
            'high_correlation': "Strong correlation detected between {feature1} and {feature2} (r={correlation:.3f})",
            'skewed_distribution': "{feature} shows {direction} skewness, consider transformation",
            'missing_data': "{feature} has {percentage}% missing values, imputation recommended",
            'outliers': "{feature} contains {percentage}% outliers, investigation recommended"
        }
    
    # Helper methods
    def _calculate_outlier_percentage(self, data: pd.Series) -> float:
        """Calculate percentage of outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        return (outliers / len(data)) * 100
    
    def _interpret_distribution(self, skewness: float, kurtosis: float) -> str:
        """Interpret distribution characteristics"""
        if abs(skewness) < 0.5:
            skew_desc = "approximately symmetric"
        elif skewness > 0.5:
            skew_desc = "right-skewed (positive skew)"
        else:
            skew_desc = "left-skewed (negative skew)"
        
        if kurtosis > 3:
            kurt_desc = "heavy-tailed (leptokurtic)"
        elif kurtosis < 3:
            kurt_desc = "light-tailed (platykurtic)"
        else:
            kurt_desc = "normal tails (mesokurtic)"
        
        return f"Distribution is {skew_desc} and {kurt_desc}"
    
    def _get_quality_grade(self, score: float) -> str:
        """Get quality grade based on score"""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        else:
            return "D"
    
    def _get_quality_recommendations(self, issues: List[str]) -> List[str]:
        """Get recommendations based on quality issues"""
        recommendations = []
        for issue in issues:
            if "missing" in issue.lower():
                recommendations.append("Implement data imputation strategies")
            elif "diversity" in issue.lower():
                recommendations.append("Consider feature engineering or removal")
            elif "constant" in issue.lower():
                recommendations.append("Remove constant features")
        return list(set(recommendations))
    
    def _interpret_correlations(self, correlations: List[Dict]) -> List[str]:
        """Interpret correlation insights"""
        insights = []
        if len(correlations) > 5:
            insights.append("High multicollinearity detected - consider feature selection")
        for corr in correlations:
            if abs(corr['correlation']) > 0.9:
                insights.append(f"Very strong correlation between {corr['feature_1']} and {corr['feature_2']} - potential redundancy")
        return insights
    
    def _calculate_trend_direction(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        if len(series) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(series))
        slope = np.polyfit(x, series.dropna(), 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_trend_strength(self, series: pd.Series) -> str:
        """Calculate trend strength"""
        if len(series) < 2:
            return "unknown"
        
        # Calculate R² for trend strength
        x = np.arange(len(series))
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return "unknown"
        
        correlation = np.corrcoef(x[:len(clean_series)], clean_series)[0, 1]
        r_squared = correlation ** 2
        
        if r_squared > 0.8:
            return "strong"
        elif r_squared > 0.5:
            return "moderate"
        else:
            return "weak"
    
    def _analyze_target_variable(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Analyze target variable patterns"""
        insights = []
        target = df[target_column]
        
        if target.dtype in [np.number]:
            insights.append(f"Target variable range: {target.min():.2f} to {target.max():.2f}")
            insights.append(f"Target variable mean: {target.mean():.2f}")
        else:
            value_counts = target.value_counts()
            insights.append(f"Target variable has {len(value_counts)} unique values")
            if len(value_counts) <= 10:
                insights.append(f"Most common value: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences)")
        
        return insights

# Dashboard integration function
def create_ai_insights_dashboard(df: pd.DataFrame, target_column: Optional[str] = None):
    """
    Create AI insights dashboard for Streamlit integration
    """
    import streamlit as st
    
    st.header("🤖 AI-Powered Data Insights")
    
    # Initialize AI insights generator
    ai_generator = AIInsightsGenerator()
    
    with st.spinner("🧠 Generating AI insights..."):
        insights = ai_generator.generate_comprehensive_insights(df, target_column)
    
    if 'error' in insights:
        st.error(f"Error generating insights: {insights['error']}")
        return
    
    # Display insights in organized tabs
    insight_tabs = st.tabs([
        "📊 Overview", 
        "📈 Statistical", 
        "🔍 Quality", 
        "🔗 Correlations", 
        "📅 Trends", 
        "💼 Business", 
        "🎯 Recommendations"
    ])
    
    with insight_tabs[0]:
        st.subheader("📊 Data Overview")
        overview = insights['data_overview']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Size", overview['dataset_size'])
            st.metric("Memory Usage", overview['memory_usage'])
        with col2:
            st.metric("Numeric Features", overview['numeric_features'])
            st.metric("Categorical Features", overview['categorical_features'])
        with col3:
            st.metric("Missing Data", overview['missing_data_percentage'])
            st.metric("Data Density", overview['data_density'])
        
        if overview['observations']:
            st.write("**🔍 Key Observations:**")
            for obs in overview['observations']:
                st.write(f"• {obs}")
    
    with insight_tabs[1]:
        st.subheader("📈 Statistical Insights")
        if 'message' in insights['statistical_insights']:
            st.info(insights['statistical_insights']['message'])
        else:
            for col, stats in insights['statistical_insights'].items():
                with st.expander(f"📊 {col}"):
                    st.write(f"**Distribution:** {stats['distribution_type']}")
                    st.write(f"**Interpretation:** {stats['interpretation']}")
                    st.write(f"**Outliers:** {stats['outlier_percentage']:.1f}%")
    
    with insight_tabs[2]:
        st.subheader("🔍 Data Quality Assessment")
        quality = insights['quality_insights']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Quality Score", f"{quality['quality_score']:.1f}/100")
        with col2:
            st.metric("Quality Grade", quality['quality_grade'])
        
        if quality['issues']:
            st.write("**⚠️ Quality Issues:**")
            for issue in quality['issues']:
                st.write(f"• {issue}")
        
        if quality['recommendations']:
            st.write("**💡 Quality Recommendations:**")
            for rec in quality['recommendations']:
                st.write(f"• {rec}")
    
    with insight_tabs[3]:
        st.subheader("🔗 Correlation Insights")
        corr_insights = insights['correlation_insights']
        if 'message' in corr_insights:
            st.info(corr_insights['message'])
        else:
            if corr_insights['strong_correlations']:
                st.write("**🔗 Strong Correlations:**")
                for corr in corr_insights['strong_correlations']:
                    st.write(f"• {corr['feature_1']} ↔ {corr['feature_2']}: {corr['correlation']:.3f} ({corr['strength']})")
            
            if corr_insights['insights']:
                st.write("**💡 Correlation Insights:**")
                for insight in corr_insights['insights']:
                    st.write(f"• {insight}")
    
    with insight_tabs[4]:
        st.subheader("📅 Trend Analysis")
        trends = insights['trend_insights']
        if 'message' in trends:
            st.info(trends['message'])
        else:
            for trend_name, trend_data in trends.items():
                st.write(f"**{trend_name}:** {trend_data['direction']} ({trend_data['strength']} trend)")
    
    with insight_tabs[5]:
        st.subheader("💼 Business Insights")
        business = insights['business_insights']['insights']
        if business:
            for insight in business:
                st.write(f"• {insight}")
        else:
            st.info("No business-specific patterns detected")
    
    with insight_tabs[6]:
        st.subheader("🎯 AI Recommendations")
        recommendations = insights['recommendations']
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.success("No specific recommendations - data appears well-structured!")
        
        # AI Narratives
        st.write("**📖 AI-Generated Narrative:**")
        narratives = insights['ai_narratives']
        st.write(narratives['dataset_story'])
        st.write(narratives['quality_narrative'])
