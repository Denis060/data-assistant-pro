"""
Model Performance Diagnostic Tool
Identifies common causes of poor ML model performance and provides actionable solutions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report
import warnings
warnings.filterwarnings('ignore')

class ModelPerformanceDiagnostic:
    """
    Comprehensive diagnostic tool to identify and fix poor model performance.
    Analyzes data quality, feature engineering, model selection, and training issues.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test, problem_type='classification'):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        self.problem_type = problem_type
        self.issues_found = []
        self.solutions = []
        
    def run_full_diagnostic(self, models_dict) -> dict:
        """
        Run comprehensive diagnostic to identify performance issues.
        
        Args:
            models_dict: Dictionary of trained models {name: model}
            
        Returns:
            Dict with diagnostic results and solutions
        """
        
        diagnostic_results = {
            'data_issues': self._diagnose_data_quality(),
            'feature_issues': self._diagnose_feature_problems(),
            'model_issues': self._diagnose_model_problems(models_dict),
            'training_issues': self._diagnose_training_problems(models_dict),
            'overall_recommendations': [],
            'priority_fixes': []
        }
        
        # Compile overall recommendations
        all_issues = []
        all_issues.extend(diagnostic_results['data_issues'])
        all_issues.extend(diagnostic_results['feature_issues'])
        all_issues.extend(diagnostic_results['model_issues'])
        all_issues.extend(diagnostic_results['training_issues'])
        
        # Sort by severity and impact
        high_priority = [issue for issue in all_issues if issue.get('severity') == 'High']
        medium_priority = [issue for issue in all_issues if issue.get('severity') == 'Medium']
        
        diagnostic_results['priority_fixes'] = high_priority[:3]  # Top 3 critical fixes
        diagnostic_results['overall_recommendations'] = high_priority + medium_priority
        
        return diagnostic_results
    
    def _diagnose_data_quality(self) -> list:
        """Identify data quality issues that hurt performance."""
        issues = []
        
        # Check for data leakage
        if self._detect_data_leakage():
            issues.append({
                'type': 'Data Leakage',
                'severity': 'High',
                'description': 'Suspiciously high correlation between features and target',
                'impact': 'Overfitting, poor real-world performance',
                'solution': 'Remove highly correlated features (>0.95 with target)',
                'fix_code': 'Remove features with correlation > 0.95 with target variable'
            })
        
        # Check train-test distribution drift
        drift_score = self._detect_distribution_drift()
        if drift_score > 0.3:
            issues.append({
                'type': 'Data Drift',
                'severity': 'High',
                'description': f'Training and test data have different distributions (drift score: {drift_score:.2f})',
                'impact': 'Model trained on different data than it will see in production',
                'solution': 'Ensure train/test split is random and representative',
                'fix_code': 'Use stratified split for classification, check for temporal ordering'
            })
        
        # Check for insufficient data
        sample_ratio = len(self.X_train) / len(self.X_train.columns)
        if sample_ratio < 10:
            issues.append({
                'type': 'Insufficient Data',
                'severity': 'High',
                'description': f'Only {sample_ratio:.1f} samples per feature (recommended: >10)',
                'impact': 'Overfitting, poor generalization',
                'solution': 'Collect more data or reduce features using feature selection',
                'fix_code': 'Aim for at least 10 samples per feature, or use dimensionality reduction'
            })
        
        # Check for class imbalance (classification only)
        if self.problem_type == 'classification':
            class_distribution = pd.Series(self.y_train).value_counts(normalize=True)
            min_class_pct = class_distribution.min()
            if min_class_pct < 0.05:  # Less than 5% representation
                issues.append({
                    'type': 'Severe Class Imbalance',
                    'severity': 'High',
                    'description': f'Minority class represents only {min_class_pct:.1%} of data',
                    'impact': 'Model biased toward majority class, poor minority class prediction',
                    'solution': 'Use SMOTE, class weights, or ensemble methods',
                    'fix_code': 'Apply SMOTE oversampling or use class_weight="balanced" in models'
                })
        
        return issues
    
    def _diagnose_feature_problems(self) -> list:
        """Identify feature engineering and selection issues."""
        issues = []
        
        # Check for irrelevant features
        irrelevant_features = self._find_irrelevant_features()
        if len(irrelevant_features) > 0:
            issues.append({
                'type': 'Irrelevant Features',
                'severity': 'Medium',
                'description': f'{len(irrelevant_features)} features have very low predictive power',
                'impact': 'Noise in data, overfitting, poor performance',
                'solution': 'Remove low-importance features using feature selection',
                'fix_code': f'Remove features: {irrelevant_features[:5]}...',
                'affected_features': irrelevant_features
            })
        
        # Check for high cardinality categorical features
        high_cardinality = self._find_high_cardinality_features()
        if len(high_cardinality) > 0:
            issues.append({
                'type': 'High Cardinality Categories',
                'severity': 'Medium',
                'description': f'{len(high_cardinality)} categorical features have too many unique values',
                'impact': 'Overfitting, curse of dimensionality',
                'solution': 'Group rare categories, use target encoding, or feature hashing',
                'fix_code': 'Group categories with <1% frequency into "Other"',
                'affected_features': high_cardinality
            })
        
        # Check for feature scaling issues
        if self._needs_feature_scaling():
            issues.append({
                'type': 'Feature Scaling',
                'severity': 'Medium',
                'description': 'Features have very different scales',
                'impact': 'Poor performance for distance-based algorithms (SVM, KNN)',
                'solution': 'Apply StandardScaler or MinMaxScaler to numeric features',
                'fix_code': 'Use StandardScaler() on numeric features before training'
            })
        
        # Check for multicollinearity
        multicollinear_features = self._detect_multicollinearity()
        if len(multicollinear_features) > 0:
            issues.append({
                'type': 'Multicollinearity',
                'severity': 'Medium',
                'description': f'{len(multicollinear_features)} features are highly correlated with each other',
                'impact': 'Unstable model coefficients, reduced interpretability',
                'solution': 'Remove one feature from each correlated pair',
                'fix_code': f'Remove features: {multicollinear_features[:3]}...',
                'affected_features': multicollinear_features
            })
        
        return issues
    
    def _diagnose_model_problems(self, models_dict) -> list:
        """Identify model selection and configuration issues."""
        issues = []
        
        # Check if using wrong model type
        if self._wrong_model_for_problem(models_dict):
            issues.append({
                'type': 'Inappropriate Model Choice',
                'severity': 'High',
                'description': 'Selected models may not be suitable for this problem',
                'impact': 'Fundamentally poor performance',
                'solution': 'Try ensemble methods (Random Forest, Gradient Boosting) for tabular data',
                'fix_code': 'Use RandomForestClassifier/Regressor or GradientBoostingClassifier/Regressor'
            })
        
        # Check for hyperparameter issues
        if self._needs_hyperparameter_tuning(models_dict):
            issues.append({
                'type': 'Default Hyperparameters',
                'severity': 'Medium',
                'description': 'Using default hyperparameters without tuning',
                'impact': 'Suboptimal performance, models not adapted to your data',
                'solution': 'Perform grid search or random search for hyperparameter tuning',
                'fix_code': 'Use GridSearchCV or RandomizedSearchCV for hyperparameter optimization'
            })
        
        return issues
    
    def _diagnose_training_problems(self, models_dict) -> list:
        """Identify training process issues."""
        issues = []
        
        # Check for overfitting
        overfitting_models = self._detect_overfitting(models_dict)
        if len(overfitting_models) > 0:
            issues.append({
                'type': 'Overfitting',
                'severity': 'High',
                'description': f'{len(overfitting_models)} models show signs of overfitting',
                'impact': 'Poor generalization to new data',
                'solution': 'Use regularization, reduce model complexity, or collect more data',
                'fix_code': 'Add regularization parameters or use simpler models',
                'affected_models': overfitting_models
            })
        
        # Check for underfitting
        underfitting_models = self._detect_underfitting(models_dict)
        if len(underfitting_models) > 0:
            issues.append({
                'type': 'Underfitting',
                'severity': 'High',
                'description': f'{len(underfitting_models)} models are too simple for the data',
                'impact': 'Unable to capture underlying patterns',
                'solution': 'Use more complex models, add features, or reduce regularization',
                'fix_code': 'Try ensemble methods or increase model complexity',
                'affected_models': underfitting_models
            })
        
        return issues
    
    def _detect_data_leakage(self) -> bool:
        """Detect potential data leakage by checking correlations."""
        try:
            # Calculate correlations between features and target
            if self.problem_type == 'classification':
                # Use label encoding for categorical targets
                if self.y_train.dtype == 'object':
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(self.y_train.astype(str))
                else:
                    y_encoded = self.y_train
            else:
                y_encoded = self.y_train
            
            correlations = []
            for col in self.X_train.select_dtypes(include=[np.number]).columns:
                corr = np.corrcoef(self.X_train[col].fillna(0), y_encoded)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            # If any feature has >95% correlation with target, it's suspicious
            return max(correlations) > 0.95 if correlations else False
        except:
            return False
    
    def _detect_distribution_drift(self) -> float:
        """Detect distribution drift between train and test sets."""
        try:
            # Calculate KS statistic for numeric features
            from scipy.stats import ks_2samp
            
            drift_scores = []
            for col in self.X_train.select_dtypes(include=[np.number]).columns:
                if col in self.X_test.columns:
                    train_vals = self.X_train[col].dropna()
                    test_vals = self.X_test[col].dropna()
                    if len(train_vals) > 0 and len(test_vals) > 0:
                        ks_stat, _ = ks_2samp(train_vals, test_vals)
                        drift_scores.append(ks_stat)
            
            return np.mean(drift_scores) if drift_scores else 0
        except:
            return 0
    
    def _find_irrelevant_features(self) -> list:
        """Find features with very low predictive power."""
        try:
            # Prepare data for feature selection
            X_numeric = self.X_train.select_dtypes(include=[np.number]).fillna(0)
            if len(X_numeric.columns) == 0:
                return []
            
            if self.problem_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k='all')
                y_target = LabelEncoder().fit_transform(self.y_train.astype(str)) if self.y_train.dtype == 'object' else self.y_train
            else:
                selector = SelectKBest(score_func=f_regression, k='all')
                y_target = self.y_train
            
            selector.fit(X_numeric, y_target)
            scores = selector.scores_
            
            # Find features with very low scores (bottom 20% or score < 1)
            threshold = max(1, np.percentile(scores, 20))
            irrelevant_idx = np.where(scores < threshold)[0]
            
            return X_numeric.columns[irrelevant_idx].tolist()
        except:
            return []
    
    def _find_high_cardinality_features(self) -> list:
        """Find categorical features with too many unique values."""
        high_cardinality = []
        
        for col in self.X_train.select_dtypes(include=['object']).columns:
            unique_count = self.X_train[col].nunique()
            # High cardinality if >50 unique values or >10% of total samples
            if unique_count > 50 or unique_count > len(self.X_train) * 0.1:
                high_cardinality.append(col)
        
        return high_cardinality
    
    def _needs_feature_scaling(self) -> bool:
        """Check if features need scaling."""
        numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return False
        
        # Calculate range of each feature
        ranges = []
        for col in numeric_cols:
            col_range = self.X_train[col].max() - self.X_train[col].min()
            if col_range > 0:
                ranges.append(col_range)
        
        if len(ranges) < 2:
            return False
        
        # If the ratio of max to min range is > 100, scaling is needed
        return (max(ranges) / min(ranges)) > 100
    
    def _detect_multicollinearity(self) -> list:
        """Detect highly correlated features."""
        try:
            numeric_cols = self.X_train.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return []
            
            corr_matrix = self.X_train[numeric_cols].corr().abs()
            
            # Find pairs with correlation > 0.9
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        high_corr_pairs.append(corr_matrix.columns[j])  # Remove the second feature
            
            return list(set(high_corr_pairs))
        except:
            return []
    
    def _wrong_model_for_problem(self, models_dict) -> bool:
        """Check if model choice is inappropriate."""
        # Simple heuristic: if using only linear models for complex non-linear data
        linear_models = ['Linear Regression', 'Logistic Regression']
        
        if all(name in linear_models for name in models_dict.keys()):
            # Check if data has non-linear patterns (this is simplified)
            return len(self.X_train.columns) > 10  # If many features, likely non-linear
        
        return False
    
    def _needs_hyperparameter_tuning(self, models_dict) -> bool:
        """Check if models are using default hyperparameters."""
        # This is a simplified check - in practice, you'd examine model parameters
        return True  # Most users don't tune hyperparameters
    
    def _detect_overfitting(self, models_dict) -> list:
        """Detect overfitting by comparing train and validation performance."""
        overfitting_models = []
        
        try:
            for name, model in models_dict.items():
                # Calculate training accuracy/score
                if self.problem_type == 'classification':
                    train_score = accuracy_score(self.y_train, model.predict(self.X_train))
                    test_score = accuracy_score(self.y_test, model.predict(self.X_test))
                else:
                    train_score = r2_score(self.y_train, model.predict(self.X_train))
                    test_score = r2_score(self.y_test, model.predict(self.X_test))
                
                # If training score is much higher than test score, it's overfitting
                if train_score - test_score > 0.2:  # 20% difference
                    overfitting_models.append(name)
        except:
            pass
        
        return overfitting_models
    
    def _detect_underfitting(self, models_dict) -> list:
        """Detect underfitting by checking if performance is very low."""
        underfitting_models = []
        
        try:
            for name, model in models_dict.items():
                if self.problem_type == 'classification':
                    test_score = accuracy_score(self.y_test, model.predict(self.X_test))
                    # If accuracy is barely better than random guessing
                    n_classes = len(np.unique(self.y_train))
                    random_accuracy = 1.0 / n_classes
                    if test_score < random_accuracy + 0.1:  # Only 10% better than random
                        underfitting_models.append(name)
                else:
                    test_score = r2_score(self.y_test, model.predict(self.X_test))
                    if test_score < 0.3:  # R² less than 0.3 is quite poor
                        underfitting_models.append(name)
        except:
            pass
        
        return underfitting_models


def create_performance_improvement_plan(diagnostic_results) -> dict:
    """
    Create a step-by-step plan to improve model performance.
    
    Args:
        diagnostic_results: Results from ModelPerformanceDiagnostic
        
    Returns:
        Dict with prioritized improvement plan
    """
    
    improvement_plan = {
        'immediate_fixes': [],
        'short_term_improvements': [],
        'long_term_strategies': [],
        'expected_improvement': 0
    }
    
    # Categorize fixes by effort and impact
    for issue in diagnostic_results['overall_recommendations']:
        severity = issue.get('severity', 'Low')
        issue_type = issue.get('type', '')
        
        if severity == 'High':
            if issue_type in ['Data Leakage', 'Severe Class Imbalance', 'Overfitting']:
                improvement_plan['immediate_fixes'].append({
                    'action': issue['solution'],
                    'code': issue['fix_code'],
                    'expected_gain': '20-50% improvement',
                    'effort': 'Low'
                })
            else:
                improvement_plan['short_term_improvements'].append({
                    'action': issue['solution'],
                    'code': issue['fix_code'],
                    'expected_gain': '10-30% improvement',
                    'effort': 'Medium'
                })
        else:
            improvement_plan['long_term_strategies'].append({
                'action': issue['solution'],
                'code': issue['fix_code'],
                'expected_gain': '5-15% improvement',
                'effort': 'High'
            })
    
    # Calculate expected overall improvement
    immediate_impact = len(improvement_plan['immediate_fixes']) * 0.35
    short_term_impact = len(improvement_plan['short_term_improvements']) * 0.20
    long_term_impact = len(improvement_plan['long_term_strategies']) * 0.10
    
    improvement_plan['expected_improvement'] = min(100, 
        (immediate_impact + short_term_impact + long_term_impact) * 100)
    
    return improvement_plan
