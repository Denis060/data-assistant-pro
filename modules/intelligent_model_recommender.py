"""
Intelligent Model Recommender
The final missing piece - intelligent model recommendation based on data characteristics
Provides smart model selection recommendations and performance predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class IntelligentModelRecommender:
    """
    Intelligent system that analyzes your data and recommends the best models
    Based on data characteristics, problem complexity, and performance requirements
    """
    
    def __init__(self):
        self.data_analysis = {}
        self.recommendations = {}
        self.analysis_log = []
        
    def analyze_data_characteristics(self, X, y, problem_type='auto'):
        """
        Comprehensive data analysis to understand characteristics
        """
        
        self.analysis_log.append("🔍 Analyzing data characteristics...")
        
        # Basic data properties
        n_samples, n_features = X.shape
        
        # Feature analysis
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Data complexity analysis
        feature_correlations = X[numeric_features].corr().abs() if len(numeric_features) > 1 else pd.DataFrame()
        high_correlations = (feature_correlations > 0.8).sum().sum() - len(feature_correlations)
        
        # Missing values
        missing_percentage = (X.isnull().sum().sum() / (n_samples * n_features)) * 100
        
        # Target analysis
        if problem_type == 'auto':
            unique_targets = len(np.unique(y))
            is_numeric_target = pd.api.types.is_numeric_dtype(y)
            problem_type = 'regression' if (is_numeric_target and unique_targets > 10) else 'classification'
        
        # Class balance (for classification)
        class_balance_ratio = 1.0
        if problem_type == 'classification':
            class_counts = pd.Series(y).value_counts()
            class_balance_ratio = class_counts.max() / class_counts.min()
        
        # Data size category
        if n_samples < 100:
            size_category = 'very_small'
        elif n_samples < 1000:
            size_category = 'small'
        elif n_samples < 10000:
            size_category = 'medium'
        elif n_samples < 100000:
            size_category = 'large'
        else:
            size_category = 'very_large'
        
        # Feature complexity
        if n_features < 5:
            feature_complexity = 'simple'
        elif n_features < 20:
            feature_complexity = 'medium'
        elif n_features < 100:
            feature_complexity = 'complex'
        else:
            feature_complexity = 'very_complex'
        
        # Noise estimation (using simple heuristic)
        noise_level = 'low'
        if len(numeric_features) > 0:
            # Estimate noise by looking at feature variance
            normalized_vars = X[numeric_features].var() / X[numeric_features].mean()
            avg_cv = normalized_vars.mean()
            
            if avg_cv > 2:
                noise_level = 'high'
            elif avg_cv > 1:
                noise_level = 'medium'
        
        self.data_analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'numeric_features': len(numeric_features),
            'categorical_features': len(categorical_features),
            'missing_percentage': missing_percentage,
            'problem_type': problem_type,
            'class_balance_ratio': class_balance_ratio,
            'high_correlations': high_correlations,
            'size_category': size_category,
            'feature_complexity': feature_complexity,
            'noise_level': noise_level
        }
        
        self.analysis_log.append(f"✅ Data Analysis Complete:")
        self.analysis_log.append(f"   - {n_samples:,} samples, {n_features} features")
        self.analysis_log.append(f"   - Problem type: {problem_type}")
        self.analysis_log.append(f"   - Size category: {size_category}")
        self.analysis_log.append(f"   - Feature complexity: {feature_complexity}")
        self.analysis_log.append(f"   - Noise level: {noise_level}")
        
        return self.data_analysis
    
    def generate_model_recommendations(self):
        """
        Generate intelligent model recommendations based on data analysis
        """
        
        if not self.data_analysis:
            return "Please run data analysis first!"
        
        analysis = self.data_analysis
        recommendations = []
        
        # Get recommendations based on data characteristics
        primary_models = self._get_primary_recommendations(analysis)
        secondary_models = self._get_secondary_recommendations(analysis)
        ensemble_recommendations = self._get_ensemble_recommendations(analysis)
        
        self.recommendations = {
            'primary': primary_models,
            'secondary': secondary_models,
            'ensembles': ensemble_recommendations,
            'reasoning': self._generate_reasoning(analysis)
        }
        
        return self.recommendations
    
    def _get_primary_recommendations(self, analysis):
        """Get primary model recommendations based on data characteristics"""
        
        primary = []
        
        problem_type = analysis['problem_type']
        size_category = analysis['size_category']
        feature_complexity = analysis['feature_complexity']
        noise_level = analysis['noise_level']
        class_balance_ratio = analysis['class_balance_ratio']
        
        # Size-based recommendations
        if size_category in ['very_small', 'small']:
            # Small datasets - prefer simple models
            if problem_type == 'classification':
                primary.extend([
                    {'model': 'Logistic Regression', 'confidence': 0.9, 'reason': 'Excellent for small datasets, interpretable'},
                    {'model': 'K-Nearest Neighbors', 'confidence': 0.8, 'reason': 'Works well with limited data'},
                    {'model': 'Naive Bayes', 'confidence': 0.7, 'reason': 'Robust with small samples'}
                ])
            else:
                primary.extend([
                    {'model': 'Linear Regression', 'confidence': 0.9, 'reason': 'Perfect for small datasets'},
                    {'model': 'Ridge Regression', 'confidence': 0.8, 'reason': 'Handles overfitting in small data'},
                    {'model': 'K-Nearest Neighbors', 'confidence': 0.7, 'reason': 'Good for small datasets'}
                ])
        
        elif size_category == 'medium':
            # Medium datasets - tree-based models excel
            if problem_type == 'classification':
                primary.extend([
                    {'model': 'Random Forest', 'confidence': 0.95, 'reason': 'Excellent balance of performance and robustness'},
                    {'model': 'Gradient Boosting', 'confidence': 0.9, 'reason': 'High accuracy, handles complex patterns'},
                    {'model': 'SVM', 'confidence': 0.8, 'reason': 'Strong performance on medium-sized datasets'}
                ])
            else:
                primary.extend([
                    {'model': 'Random Forest', 'confidence': 0.95, 'reason': 'Top choice for regression with medium data'},
                    {'model': 'Gradient Boosting', 'confidence': 0.9, 'reason': 'Excellent for complex relationships'},
                    {'model': 'SVM', 'confidence': 0.8, 'reason': 'Robust for regression tasks'}
                ])
        
        else:  # large or very_large
            # Large datasets - ensemble methods and deep learning
            if problem_type == 'classification':
                primary.extend([
                    {'model': 'Gradient Boosting', 'confidence': 0.95, 'reason': 'Scales well, highest accuracy potential'},
                    {'model': 'Random Forest', 'confidence': 0.9, 'reason': 'Robust and parallelizable'},
                    {'model': 'Logistic Regression', 'confidence': 0.8, 'reason': 'Fast training on large data'}
                ])
            else:
                primary.extend([
                    {'model': 'Gradient Boosting', 'confidence': 0.95, 'reason': 'Best for large-scale regression'},
                    {'model': 'Random Forest', 'confidence': 0.9, 'reason': 'Handles large datasets well'},
                    {'model': 'Linear Regression', 'confidence': 0.8, 'reason': 'Efficient for large data'}
                ])
        
        # Adjust for class imbalance
        if problem_type == 'classification' and class_balance_ratio > 3:
            # Heavily imbalanced - adjust recommendations
            for rec in primary:
                if rec['model'] in ['Random Forest', 'Gradient Boosting']:
                    rec['confidence'] *= 1.1  # Boost tree-based models
                    rec['reason'] += ' (handles imbalanced data well)'
        
        # Adjust for high feature complexity
        if feature_complexity in ['complex', 'very_complex']:
            for rec in primary:
                if rec['model'] in ['Random Forest', 'Gradient Boosting']:
                    rec['confidence'] *= 1.1  # Boost ensemble methods
                    rec['reason'] += ' (excellent with many features)'
        
        # Sort by confidence
        primary.sort(key=lambda x: x['confidence'], reverse=True)
        
        return primary[:3]  # Return top 3
    
    def _get_secondary_recommendations(self, analysis):
        """Get secondary/alternative model recommendations"""
        
        secondary = []
        
        problem_type = analysis['problem_type']
        noise_level = analysis['noise_level']
        
        # Add models that might work well as alternatives
        if problem_type == 'classification':
            secondary.extend([
                {'model': 'Decision Tree', 'confidence': 0.6, 'reason': 'Interpretable, good for feature understanding'},
                {'model': 'AdaBoost', 'confidence': 0.7, 'reason': 'Good ensemble alternative'},
                {'model': 'Extra Trees', 'confidence': 0.8, 'reason': 'Fast random forest variant'}
            ])
        else:
            secondary.extend([
                {'model': 'Decision Tree', 'confidence': 0.6, 'reason': 'Simple and interpretable'},
                {'model': 'ElasticNet', 'confidence': 0.7, 'reason': 'Good for feature selection'},
                {'model': 'Lasso Regression', 'confidence': 0.7, 'reason': 'Automatic feature selection'}
            ])
        
        # Adjust for noise
        if noise_level == 'high':
            for rec in secondary:
                if rec['model'] in ['Random Forest', 'Gradient Boosting']:
                    rec['confidence'] *= 1.1
                    rec['reason'] += ' (robust to noise)'
        
        secondary.sort(key=lambda x: x['confidence'], reverse=True)
        return secondary[:2]  # Return top 2
    
    def _get_ensemble_recommendations(self, analysis):
        """Get ensemble method recommendations"""
        
        ensembles = []
        
        size_category = analysis['size_category']
        
        if size_category in ['medium', 'large', 'very_large']:
            ensembles.extend([
                {'model': 'Voting Ensemble', 'confidence': 0.9, 'reason': 'Combines multiple models for better performance'},
                {'model': 'Stacking Ensemble', 'confidence': 0.85, 'reason': 'Advanced ensemble with meta-learning'},
                {'model': 'Bagging Ensemble', 'confidence': 0.8, 'reason': 'Reduces overfitting through variance reduction'}
            ])
        else:
            ensembles.append({
                'model': 'Simple Voting', 'confidence': 0.7, 'reason': 'Basic ensemble suitable for smaller datasets'
            })
        
        return ensembles
    
    def _generate_reasoning(self, analysis):
        """Generate detailed reasoning for recommendations"""
        
        reasoning = []
        
        # Data size reasoning
        size_category = analysis['size_category']
        n_samples = analysis['n_samples']
        
        if size_category == 'very_small':
            reasoning.append(f"🔸 **Small Dataset** ({n_samples:,} samples): Simple models recommended to avoid overfitting")
        elif size_category == 'small':
            reasoning.append(f"🔸 **Small-Medium Dataset** ({n_samples:,} samples): Linear models and KNN work well")
        elif size_category == 'medium':
            reasoning.append(f"🔸 **Medium Dataset** ({n_samples:,} samples): Tree-based models (RF, GB) are optimal")
        else:
            reasoning.append(f"🔸 **Large Dataset** ({n_samples:,} samples): Advanced ensembles and boosting excel")
        
        # Feature complexity reasoning
        feature_complexity = analysis['feature_complexity']
        n_features = analysis['n_features']
        
        if feature_complexity == 'simple':
            reasoning.append(f"🔸 **Simple Features** ({n_features} features): Linear models provide good interpretability")
        elif feature_complexity == 'medium':
            reasoning.append(f"🔸 **Medium Complexity** ({n_features} features): Tree-based models handle interactions well")
        else:
            reasoning.append(f"🔸 **High Complexity** ({n_features} features): Ensemble methods manage complexity best")
        
        # Class balance reasoning
        if analysis['problem_type'] == 'classification':
            class_balance_ratio = analysis['class_balance_ratio']
            if class_balance_ratio > 3:
                reasoning.append(f"🔸 **Imbalanced Classes** (ratio: {class_balance_ratio:.1f}): Tree-based models with class weights recommended")
        
        # Noise level reasoning
        noise_level = analysis['noise_level']
        if noise_level == 'high':
            reasoning.append("🔸 **High Noise Level**: Robust models (RF, GB) recommended over sensitive models")
        elif noise_level == 'low':
            reasoning.append("🔸 **Low Noise Level**: High-precision models can achieve excellent performance")
        
        return reasoning
    
    def predict_model_performance(self, X, y, recommended_models):
        """
        Predict expected performance for recommended models
        """
        
        performance_predictions = {}
        
        self.analysis_log.append("🎯 Predicting model performance...")
        
        # Use meta-learning approach based on data characteristics
        analysis = self.data_analysis
        
        for model_rec in recommended_models:
            model_name = model_rec['model']
            
            # Base performance estimate based on data characteristics
            base_performance = self._estimate_base_performance(analysis)
            
            # Model-specific adjustments
            model_adjustment = self._get_model_performance_adjustment(model_name, analysis)
            
            # Final prediction
            predicted_performance = base_performance * model_adjustment
            
            performance_predictions[model_name] = {
                'predicted_score': predicted_performance,
                'confidence_interval': (predicted_performance * 0.9, predicted_performance * 1.1),
                'recommendation_confidence': model_rec['confidence']
            }
        
        return performance_predictions
    
    def _estimate_base_performance(self, analysis):
        """Estimate base performance level based on data characteristics"""
        
        # Start with problem-type baseline
        if analysis['problem_type'] == 'classification':
            base = 0.7  # 70% accuracy baseline
        else:
            base = 0.6  # 60% R² baseline
        
        # Adjust for data quality factors
        if analysis['missing_percentage'] < 5:
            base *= 1.1  # Clean data boost
        elif analysis['missing_percentage'] > 20:
            base *= 0.9  # Dirty data penalty
        
        # Adjust for size
        if analysis['size_category'] in ['large', 'very_large']:
            base *= 1.1  # More data = better performance
        elif analysis['size_category'] == 'very_small':
            base *= 0.9  # Less data = harder to learn
        
        # Adjust for feature quality
        if analysis['feature_complexity'] == 'simple':
            base *= 1.05  # Simple problems are easier
        elif analysis['feature_complexity'] == 'very_complex':
            base *= 0.95  # Complex problems are harder
        
        return min(base, 0.95)  # Cap at 95%
    
    def _get_model_performance_adjustment(self, model_name, analysis):
        """Get model-specific performance adjustments"""
        
        adjustments = {
            'Random Forest': 1.1,      # Generally strong
            'Gradient Boosting': 1.15, # Often best performer
            'SVM': 1.05,              # Solid performance
            'Logistic Regression': 1.0, # Baseline
            'Linear Regression': 1.0,   # Baseline
            'K-Nearest Neighbors': 0.95, # Can be inconsistent
            'Naive Bayes': 0.9,       # Often lower performance
            'Decision Tree': 0.85      # Prone to overfitting
        }
        
        base_adjustment = adjustments.get(model_name, 1.0)
        
        # Context-specific adjustments
        if analysis['size_category'] == 'very_small':
            if model_name in ['Logistic Regression', 'Linear Regression']:
                base_adjustment *= 1.1  # Linear models excel with small data
            elif model_name in ['Random Forest', 'Gradient Boosting']:
                base_adjustment *= 0.9  # Complex models struggle with small data
        
        if analysis['noise_level'] == 'high':
            if model_name in ['Random Forest', 'Gradient Boosting']:
                base_adjustment *= 1.05  # Robust to noise
            elif model_name in ['K-Nearest Neighbors', 'Decision Tree']:
                base_adjustment *= 0.95  # Sensitive to noise
        
        return base_adjustment
    
    def generate_comprehensive_report(self, X, y):
        """Generate a comprehensive model recommendation report"""
        
        # Run complete analysis
        data_analysis = self.analyze_data_characteristics(X, y)
        recommendations = self.generate_model_recommendations()
        
        # Predict performance for primary recommendations
        performance_predictions = self.predict_model_performance(X, y, recommendations['primary'])
        
        report = {
            'data_analysis': data_analysis,
            'recommendations': recommendations,
            'performance_predictions': performance_predictions,
            'analysis_log': self.analysis_log,
            'summary': self._generate_summary(data_analysis, recommendations)
        }
        
        return report
    
    def _generate_summary(self, analysis, recommendations):
        """Generate executive summary of recommendations"""
        
        primary_model = recommendations['primary'][0]['model']
        confidence = recommendations['primary'][0]['confidence']
        
        summary = f"""
        ## 🎯 Model Recommendation Summary
        
        **Top Recommendation**: {primary_model} (Confidence: {confidence:.0%})
        
        **Dataset Profile**:
        - Size: {analysis['n_samples']:,} samples, {analysis['n_features']} features
        - Type: {analysis['problem_type'].title()}
        - Complexity: {analysis['feature_complexity'].title()}
        - Category: {analysis['size_category'].replace('_', ' ').title()}
        
        **Why this model**: {recommendations['primary'][0]['reason']}
        
        **Alternative Options**: {', '.join([rec['model'] for rec in recommendations['primary'][1:]])}
        """
        
        return summary
