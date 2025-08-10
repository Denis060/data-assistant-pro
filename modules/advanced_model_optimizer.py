"""
Advanced Model Optimizer
Implements specific solutions for common ML performance issues
Addresses overfitting, feature selection, and hyperparameter optimization
"""

import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class AdvancedModelOptimizer:
    """
    Advanced optimizer that implements specific solutions for:
    1. Overfitting prevention
    2. Feature selection
    3. Hyperparameter optimization
    """
    
    def __init__(self, X, y, problem_type='auto'):
        self.X = X.copy()
        self.y = y.copy()
        self.problem_type = self._detect_problem_type(y) if problem_type == 'auto' else problem_type
        self.optimization_log = []
        self.selected_features = None
        self.best_params = {}
        
        # Diagnose data issues
        self._diagnose_data_issues()
        
        # Engineer new features
        self._engineer_features()
        
        # Handle multicollinearity
        self._handle_multicollinearity()
        
    def _engineer_features(self):
        """
        Automatically engineer new features from existing ones, focusing on date/time.
        """
        self.optimization_log.append("🛠️ Engineering new features...")
        
        # Convert object columns that look like dates to datetime
        for col in self.X.select_dtypes(include=['object']).columns:
            try:
                # Attempt to convert to datetime, but don't be too aggressive
                converted = pd.to_datetime(self.X[col], errors='coerce')
                # Check if a significant portion was converted successfully
                if converted.notna().sum() / len(self.X) > 0.8:
                    self.X[col] = converted
                    self.optimization_log.append(f"  • Converted column `{col}` to datetime.")
            except Exception:
                continue # Ignore columns that can't be converted

        # Now, find datetime columns and engineer features
        date_cols = self.X.select_dtypes(include=['datetime64']).columns
        
        if not date_cols.any():
            self.optimization_log.append("  • No date/time features found to engineer.")
            return

        for col in date_cols:
            self.optimization_log.append(f"  • Engineering features from `{col}`...")
            
            # Calculate tenure in years
            # Assuming the 'present' is the most recent date in the column
            present_date = self.X[col].max()
            tenure_days = (present_date - self.X[col]).dt.days
            tenure_years = tenure_days / 365.25
            
            new_col_name = f"{col}_tenure_years"
            self.X[new_col_name] = tenure_years
            self.optimization_log.append(f"    -> Created `{new_col_name}`.")
            
            # Drop the original date column as it's no longer needed
            self.X = self.X.drop(columns=[col])
            self.optimization_log.append(f"    -> Dropped original date column `{col}`.")

        self.optimization_log.append("✅ Feature engineering complete.")

    def _handle_multicollinearity(self, threshold=0.95):
        """Detect and remove highly correlated features to reduce redundancy."""
        self.optimization_log.append(f"🔎 Checking for multicollinearity (threshold={threshold})...")
        
        # Ensure only numeric columns are used for correlation
        numeric_X = self.X.select_dtypes(include=np.number)
        corr_matrix = numeric_X.corr().abs()
        
        # Create a boolean matrix of correlations above the threshold
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        if to_drop:
            # Log the pairs for clarity
            for col in to_drop:
                correlated_with = upper.index[upper[col] > threshold].tolist()
                self.optimization_log.append(f"  • Found correlation: `{col}` with `{correlated_with[0]}`")

            self.optimization_log.append(f"🔥 High correlation detected. Removing {len(to_drop)} feature(s): {to_drop}")
            self.X = self.X.drop(columns=to_drop)
            self.optimization_log.append(f"✅ Features removed. New shape of X: {self.X.shape}")
        else:
            self.optimization_log.append("✅ No high multicollinearity found.")
        
    def diagnose_performance_issues(self, models_dict, cv_folds=5):
        """
        Diagnose performance issues by analyzing cross-validation scores.
        Returns a dictionary of issues found for each model.
        """
        self.optimization_log.append("🩺 Diagnosing model performance issues...")
        issues = {'overfitting': [], 'underfitting': [], 'severe_underfitting': []}
        scoring = 'r2' if self.problem_type == 'regression' else 'accuracy'

        for model_name, model in models_dict.items():
            try:
                scores = cross_val_score(model, self.X, self.y, cv=cv_folds, scoring=scoring)
                mean_score = np.mean(scores)
                std_dev = np.std(scores)

                # Log individual model performance
                self.optimization_log.append(f"  • {model_name}: CV Mean Score = {mean_score:.4f}, Std Dev = {std_dev:.4f}")

                # Issue detection logic
                is_underfitting = False
                if self.problem_type == 'regression':
                    if mean_score < 0.4:
                        issues['severe_underfitting'].append(model_name)
                        is_underfitting = True
                    elif mean_score < 0.7:
                        issues['underfitting'].append(model_name)
                        is_underfitting = True
                else: # Classification
                    if mean_score < 0.5:
                        issues['severe_underfitting'].append(model_name)
                        is_underfitting = True
                    elif mean_score < 0.75:
                        issues['underfitting'].append(model_name)
                        is_underfitting = True
                
                # Check for overfitting only if not severely underfitting
                # Overfitting is indicated by high variance (std_dev)
                if not is_underfitting and std_dev > 0.15:
                    issues['overfitting'].append(model_name)

            except Exception as e:
                self.optimization_log.append(f"  • ERROR diagnosing {model_name}: {e}")
        
        if not any(issues.values()):
            self.optimization_log.append("✅ All models show reasonable baseline performance.")
        else:
            if issues['severe_underfitting']:
                self.optimization_log.append(f"  -> Found SEVERE UNDERFITTING in: {issues['severe_underfitting']}")
            if issues['underfitting']:
                self.optimization_log.append(f"  -> Found moderate underfitting in: {issues['underfitting']}")
            if issues['overfitting']:
                self.optimization_log.append(f"  -> Found potential overfitting in: {issues['overfitting']}")

        return issues

    def _diagnose_data_issues(self):
        """Diagnose potential data quality issues that could cause poor performance."""
        issues = []
        
        # Check class distribution for classification
        if self.problem_type == 'classification':
            class_counts = pd.Series(self.y).value_counts()
            n_classes = len(class_counts)
            
            self.optimization_log.append(f"📊 Target Analysis: {n_classes} classes detected")
            
            # Check for too many classes
            if n_classes > 100:
                issues.append(f"ERROR: {n_classes} classes detected! This might be a regression problem mislabeled as classification.")
                
            # Check for class imbalance
            class_ratios = class_counts / len(self.y)
            min_ratio = class_ratios.min()
            max_ratio = class_ratios.max()
            
            self.optimization_log.append(f"📊 Class balance: Smallest class {min_ratio:.3%}, Largest class {max_ratio:.3%}")
            
            if min_ratio < 0.001:  # Less than 0.1% of data
                issues.append(f"CRITICAL: Severe class imbalance. Smallest class: {min_ratio:.4%}")
                
            if max_ratio > 0.99:  # More than 99% of data in one class
                issues.append(f"CRITICAL: Extreme class imbalance. Largest class: {max_ratio:.3%}")
                
            # Check if classes are actually continuous values (regression problem)
            try:
                y_numeric = pd.to_numeric(self.y, errors='coerce')
                if not y_numeric.isna().any():
                    unique_ratio = len(np.unique(self.y)) / len(self.y)
                    if unique_ratio > 0.1:  # More than 10% unique values
                        issues.append(f"WARNING: Target has many unique numeric values ({unique_ratio:.2%}). Consider regression instead.")
            except:
                pass
        
        # Check for feature issues
        n_features = self.X.shape[1]
        n_samples = self.X.shape[0]
        
        self.optimization_log.append(f"📊 Data dimensions: {n_samples} samples × {n_features} features")
        
        if n_features > n_samples:
            issues.append(f"WARNING: More features ({n_features}) than samples ({n_samples}). High risk of overfitting.")
            
        # Check for missing values
        missing_ratio = self.X.isnull().sum().sum() / (self.X.shape[0] * self.X.shape[1])
        if missing_ratio > 0.3:
            issues.append(f"WARNING: High missing value ratio: {missing_ratio:.2%}")
            
        # Check for constant features
        constant_features = []
        for col in self.X.columns:
            if self.X[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            issues.append(f"WARNING: Constant features detected: {constant_features[:5]}{'...' if len(constant_features) > 5 else ''}")
            
        # Check for ID-like columns that shouldn't be used for modeling
        id_like_features = []
        for col in self.X.columns:
            unique_ratio = self.X[col].nunique() / len(self.X)
            if unique_ratio > 0.95:  # More than 95% unique values
                id_like_features.append(col)
                
        if id_like_features:
            issues.append(f"CRITICAL: ID-like columns detected: {id_like_features[:3]}{'...' if len(id_like_features) > 3 else ''}")
            
        # Log issues
        if issues:
            self.optimization_log.append("⚠️ DATA QUALITY ISSUES DETECTED:")
            for issue in issues:
                self.optimization_log.append(f"  • {issue}")
                
            # Suggest fixes
            self.optimization_log.append("💡 SUGGESTED FIXES:")
            if any("ID-like" in issue for issue in issues):
                self.optimization_log.append("  • Remove ID columns before modeling")
            if any("class imbalance" in issue for issue in issues):
                self.optimization_log.append("  • Use class balancing techniques (SMOTE, class_weight)")
            if any("regression problem" in issue for issue in issues):
                self.optimization_log.append("  • Switch to regression problem type")
        else:
            self.optimization_log.append("✅ No major data quality issues detected")
            
        return issues
        
    def _detect_problem_type(self, y):
        """Auto-detect if this is a classification or regression problem."""
        # Check if target is string/object type
        if y.dtype == 'object':
            return 'classification'
        
        # Check number of unique values
        unique_values = len(np.unique(y))
        
        # If very few unique values, likely classification
        if unique_values <= 20:
            return 'classification'
            
        # If many unique values but all integers, might still be classification
        if unique_values <= 50 and np.all(y == y.astype(int)):
            return 'classification'
            
        # Otherwise, regression
        return 'regression'
    
    def solve_underfitting(self, models_dict, cv_folds=5):
        """
        Implement solutions for underfitting:
        1. Increase model complexity
        2. Add more features
        3. Remove regularization
        4. Check data preprocessing
        """
        
        underfitting_solutions = {}
        
        self.optimization_log.append("🎯 Implementing Underfitting Solutions...")
        
        for model_name, model in models_dict.items():
            
            # 1. Increase Model Complexity
            complex_model = self._increase_complexity(model, model_name)
            
            # 2. Remove/Reduce Regularization
            unregularized_model = self._remove_regularization(model, model_name)
            
            # 3. Cross-validation comparison
            scoring = 'r2' if self.problem_type == 'regression' else 'accuracy'
            
            original_scores = cross_val_score(model, self.X, self.y, cv=cv_folds, scoring=scoring)
            complex_scores = cross_val_score(complex_model, self.X, self.y, cv=cv_folds, scoring=scoring)
            unregularized_scores = cross_val_score(unregularized_model, self.X, self.y, cv=cv_folds, scoring=scoring)
            
            # Select best approach
            original_mean = np.mean(original_scores)
            complex_mean = np.mean(complex_scores)
            unregularized_mean = np.mean(unregularized_scores)
            
            # Choose model with best performance
            scores_comparison = [
                ('original', original_mean, model),
                ('complex', complex_mean, complex_model),
                ('unregularized', unregularized_mean, unregularized_model)
            ]
            
            # Sort by score (higher is better)
            scores_comparison.sort(key=lambda x: x[1], reverse=True)
            best_approach = scores_comparison[0]
            
            underfitting_solutions[model_name] = {
                'best_model': best_approach[2],
                'approach': best_approach[0],
                'cv_score_mean': best_approach[1],
                'improvement': best_approach[1] - original_mean
            }
            
            self.optimization_log.append(
                f"✅ {model_name}: Best approach = {best_approach[0]} "
                f"(CV Score: {best_approach[1]:.4f}, Improvement: {best_approach[1] - original_mean:.4f})"
            )
        
        return underfitting_solutions
    
    def _increase_complexity(self, model, model_name):
        """Increase model complexity to combat underfitting."""
        model_type = type(model).__name__
        
        if 'RandomForest' in model_type:
            # More trees, deeper trees
            complex_model = type(model)(
                n_estimators=300,  # More trees
                max_depth=None,    # No depth limit
                min_samples_split=2,  # Allow smaller splits
                min_samples_leaf=1,   # Allow smaller leaves
                random_state=42
            )
        
        elif 'GradientBoosting' in model_type:
            # More estimators, higher learning rate
            complex_model = type(model)(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.2,
                random_state=42
            )
        
        elif 'SVC' in model_type or 'SVR' in model_type:
            # More complex kernel, higher C
            complex_model = type(model)(kernel='rbf', C=100, gamma='scale')
        
        elif 'KNeighbors' in model_type:
            # Fewer neighbors for more complex decision boundary
            complex_model = type(model)(n_neighbors=3)
        
        else:
            # For linear models, return original (complexity handled differently)
            complex_model = model
        
        return complex_model
    
    def _remove_regularization(self, model, model_name):
        """Remove or reduce regularization to combat underfitting."""
        model_type = type(model).__name__
        
        if 'LogisticRegression' in model_type:
            # Remove regularization
            from sklearn.linear_model import LogisticRegression
            unregularized = LogisticRegression(C=1000, random_state=42, max_iter=1000)
        
        elif 'Ridge' in model_type or 'Lasso' in model_type:
            # Very low regularization
            unregularized = type(model)(alpha=0.001)
        
        elif 'SVC' in model_type or 'SVR' in model_type:
            # High C (low regularization)
            unregularized = type(model)(C=1000, kernel=model.kernel if hasattr(model, 'kernel') else 'rbf')
        
        else:
            # Return original model if no regularization to remove
            unregularized = model
        
        return unregularized
    
    def solve_overfitting(self, models_dict, cv_folds=5, test_size=0.2):
        """
        Implement overfitting solutions:
        1. Add regularization to models
        2. Reduce model complexity
        3. Use cross-validation to detect overfitting
        """
        
        overfitting_solutions = {}
        
        self.optimization_log.append("🎯 Implementing Overfitting Solutions...")
        
        for model_name, model in models_dict.items():
            
            # 1. Add Regularization
            regularized_model = self._add_regularization(model, model_name)
            
            # 2. Reduce Model Complexity
            simplified_model = self._reduce_complexity(model, model_name)
            
            # 3. Cross-validation comparison
            scoring = 'r2' if self.problem_type == 'regression' else 'accuracy'
            
            original_scores = cross_val_score(model, self.X, self.y, cv=cv_folds, scoring=scoring)
            regularized_scores = cross_val_score(regularized_model, self.X, self.y, cv=cv_folds, scoring=scoring)
            simplified_scores = cross_val_score(simplified_model, self.X, self.y, cv=cv_folds, scoring=scoring)
            
            # Select best approach
            original_mean = np.mean(original_scores)
            original_std = np.std(original_scores)
            
            regularized_mean = np.mean(regularized_scores)
            regularized_std = np.std(regularized_scores)
            
            simplified_mean = np.mean(simplified_scores)
            simplified_std = np.std(simplified_scores)
            
            # Choose model with best mean score and lowest variance (less overfitting)
            scores_comparison = [
                ('original', original_mean, original_std, model),
                ('regularized', regularized_mean, regularized_std, regularized_model),
                ('simplified', simplified_mean, simplified_std, simplified_model)
            ]
            
            # Sort by mean score (higher is better)
            scores_comparison.sort(key=lambda x: x[1], reverse=True)
            
            # If top 2 scores are close, choose the one with lower variance to combat overfitting
            if abs(scores_comparison[0][1] - scores_comparison[1][1]) < 0.05: # 5% tolerance
                best_approach = min(scores_comparison[:2], key=lambda x: x[2])
            else:
                best_approach = scores_comparison[0]
            
            overfitting_solutions[model_name] = {
                'best_model': best_approach[3],
                'approach': best_approach[0],
                'cv_score_mean': best_approach[1],
                'cv_score_std': best_approach[2],
                'overfitting_score': original_std,  # Higher std indicates more overfitting
                'improvement': best_approach[1] - original_mean
            }
            
            self.optimization_log.append(
                f"✅ {model_name}: Best approach = {best_approach[0]} "
                f"(CV Score: {best_approach[1]:.4f} ± {best_approach[2]:.4f})"
            )
        
        return overfitting_solutions
    
    def _add_regularization(self, model, model_name):
        """Add appropriate regularization based on model type."""
        model_type = type(model).__name__
        
        if 'RandomForest' in model_type:
            # Increase min_samples_split and min_samples_leaf for regularization
            regularized = type(model)(
                n_estimators=model.n_estimators,
                max_depth=min(model.max_depth or 10, 8),  # Limit depth
                min_samples_split=max(model.min_samples_split, 10),
                min_samples_leaf=max(model.min_samples_leaf, 5),
                random_state=42
            )
        
        elif 'LinearRegression' in model_type:
            # Replace with Ridge regression for regularization
            from sklearn.linear_model import Ridge
            regularized = Ridge(alpha=1.0)
        
        elif 'LogisticRegression' in model_type:
            # Add L2 regularization
            from sklearn.linear_model import LogisticRegression
            regularized = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
        
        elif 'SVC' in model_type or 'SVR' in model_type:
            # Increase regularization parameter C
            regularized = type(model)(C=0.1, kernel=model.kernel if hasattr(model, 'kernel') else 'rbf')
        
        elif 'GradientBoosting' in model_type or 'XGB' in model_type:
            # Reduce learning rate and increase regularization
            regularized = type(model)(
                learning_rate=0.05,
                n_estimators=100,
                max_depth=3,
                random_state=42
            )
        
        else:
            # Default: return original model
            regularized = model
        
        return regularized
    
    def _reduce_complexity(self, model, model_name):
        """Reduce model complexity to prevent overfitting."""
        model_type = type(model).__name__
        
        if 'RandomForest' in model_type:
            # Fewer trees, limited depth
            simplified = type(model)(
                n_estimators=50,  # Fewer trees
                max_depth=5,      # Shallow trees
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        
        elif 'GradientBoosting' in model_type or 'XGB' in model_type:
            # Fewer estimators, limited depth
            simplified = type(model)(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        
        elif 'SVC' in model_type or 'SVR' in model_type:
            # Linear kernel (simpler)
            simplified = type(model)(kernel='linear', C=1.0)
        
        else:
            # For linear models, use fewer features (will be handled in feature selection)
            simplified = model
        
        return simplified
    
    def intelligent_feature_selection(self, n_features_to_select=None, selection_methods=['univariate', 'rfe', 'model_based']):
        """
        Implement multiple feature selection methods and choose the best combination.
        """
        
        self.optimization_log.append("🔍 Starting Intelligent Feature Selection...")
        
        if n_features_to_select is None:
            # Auto-determine optimal number of features
            n_features_to_select = min(20, int(self.X.shape[1] * 0.7))
        
        feature_selection_results = {}
        
        # 1. Univariate Feature Selection
        if 'univariate' in selection_methods:
            univariate_features = self._univariate_selection(n_features_to_select)
            feature_selection_results['univariate'] = univariate_features
        
        # 2. Recursive Feature Elimination (RFE)
        if 'rfe' in selection_methods:
            rfe_features = self._rfe_selection(n_features_to_select)
            feature_selection_results['rfe'] = rfe_features
        
        # 3. Model-based Feature Selection
        if 'model_based' in selection_methods:
            model_features = self._model_based_selection(n_features_to_select)
            feature_selection_results['model_based'] = model_features
        
        # 4. Variance Threshold (remove low-variance features)
        variance_features = self._variance_threshold_selection()
        feature_selection_results['variance'] = variance_features
        
        # 5. Combine results and find consensus
        best_features = self._find_feature_consensus(feature_selection_results)
        
        # 6. Validate feature selection performance
        validation_results = self._validate_feature_selection(feature_selection_results)
        
        # Select best method based on validation
        best_method = max(validation_results.keys(), key=lambda x: validation_results[x]['score'])
        self.selected_features = feature_selection_results[best_method]['selected_features']
        
        self.optimization_log.append(
            f"✅ Selected {len(self.selected_features)} features using {best_method} method "
            f"(Performance improvement: {validation_results[best_method]['improvement']:.4f})"
        )
        
        return {
            'selected_features': self.selected_features,
            'feature_scores': feature_selection_results[best_method]['feature_scores'],
            'method_used': best_method,
            'performance_improvement': validation_results[best_method]['improvement'],
            'all_results': validation_results
        }
    
    def _univariate_selection(self, k):
        """Univariate feature selection using statistical tests."""
        if self.problem_type == 'regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        else:
            selector = SelectKBest(score_func=f_classif, k=k)
        
        X_selected = selector.fit_transform(self.X, self.y)
        selected_features = self.X.columns[selector.get_support()].tolist()
        feature_scores = dict(zip(self.X.columns, selector.scores_))
        
        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'selector': selector
        }
    
    def _rfe_selection(self, n_features):
        """Recursive Feature Elimination."""
        if self.problem_type == 'regression':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_selected = selector.fit_transform(self.X, self.y)
        selected_features = self.X.columns[selector.get_support()].tolist()
        
        # Get feature rankings
        feature_scores = dict(zip(self.X.columns, 1.0 / selector.ranking_))  # Convert ranking to score
        
        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'selector': selector
        }
    
    def _model_based_selection(self, n_features):
        """Model-based feature selection using feature importance."""
        if self.problem_type == 'regression':
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        estimator.fit(self.X, self.y)
        
        # Select features based on importance
        selector = SelectFromModel(estimator, max_features=n_features)
        X_selected = selector.fit_transform(self.X, self.y)
        selected_features = self.X.columns[selector.get_support()].tolist()
        
        feature_scores = dict(zip(self.X.columns, estimator.feature_importances_))
        
        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'selector': selector
        }
    
    def _variance_threshold_selection(self, threshold=0.01):
        """Remove features with low variance."""
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(self.X)
        selected_features = self.X.columns[selector.get_support()].tolist()
        
        feature_scores = dict(zip(self.X.columns, selector.variances_))
        
        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'selector': selector
        }
    
    def _find_feature_consensus(self, feature_results):
        """Find features that appear in multiple selection methods."""
        all_features = set()
        feature_counts = {}
        
        for method, result in feature_results.items():
            features = result['selected_features']
            all_features.update(features)
            
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Sort features by consensus (how many methods selected them)
        consensus_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'consensus_ranking': consensus_features,
            'high_consensus': [f for f, count in consensus_features if count >= 2]
        }
    
    def _validate_feature_selection(self, feature_results):
        """Validate different feature selection methods using cross-validation."""
        validation_results = {}
        
        # Use a simple model for validation
        if self.problem_type == 'regression':
            validator_model = RandomForestRegressor(n_estimators=50, random_state=42)
            scoring = 'neg_mean_squared_error'
        else:
            validator_model = RandomForestClassifier(n_estimators=50, random_state=42)
            scoring = 'accuracy'
        
        # Baseline score (all features)
        baseline_scores = cross_val_score(validator_model, self.X, self.y, cv=5, scoring=scoring)
        baseline_mean = np.mean(baseline_scores)
        
        for method, result in feature_results.items():
            selected_features = result['selected_features']
            
            if len(selected_features) > 0:
                X_selected = self.X[selected_features]
                scores = cross_val_score(validator_model, X_selected, self.y, cv=5, scoring=scoring)
                score_mean = np.mean(scores)
                improvement = score_mean - baseline_mean
                
                validation_results[method] = {
                    'score': score_mean,
                    'improvement': improvement,
                    'n_features': len(selected_features),
                    'feature_efficiency': score_mean / len(selected_features)  # Score per feature
                }
        
        return validation_results
    
    def optimize_hyperparameters(self, models_dict, search_type='grid', n_iter=50, cv_folds=5):
        """
        Comprehensive hyperparameter optimization.
        """
        
        self.optimization_log.append(f"⚙️ Starting {search_type.title()} Search Hyperparameter Optimization...")
        
        optimized_models = {}
        
        for model_name, model in models_dict.items():
            
            # Get parameter grid for this model
            param_grid = self._get_parameter_grid(model, model_name)
            
            if not param_grid:
                optimized_models[model_name] = {
                    'best_model': model,
                    'best_params': {},
                    'best_score': None,
                    'improvement': 0
                }
                continue
            
            # Perform hyperparameter search
            if search_type == 'grid':
                search = GridSearchCV(
                    model, param_grid, cv=cv_folds,
                    scoring='neg_mean_squared_error' if self.problem_type == 'regression' else 'accuracy',
                    n_jobs=-1, verbose=0
                )
            else:  # random search
                search = RandomizedSearchCV(
                    model, param_grid, n_iter=n_iter, cv=cv_folds,
                    scoring='neg_mean_squared_error' if self.problem_type == 'regression' else 'accuracy',
                    n_jobs=-1, verbose=0, random_state=42
                )
            
            # Use selected features if available
            X_train = self.X[self.selected_features] if self.selected_features else self.X
            
            # Fit the search
            search.fit(X_train, self.y)
            
            # Calculate improvement
            original_score = cross_val_score(model, X_train, self.y, cv=cv_folds,
                                           scoring='neg_mean_squared_error' if self.problem_type == 'regression' else 'accuracy')
            improvement = search.best_score_ - np.mean(original_score)
            
            optimized_models[model_name] = {
                'best_model': search.best_estimator_,
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'improvement': improvement,
                'cv_results': search.cv_results_
            }
            
            self.best_params[model_name] = search.best_params_
            
            self.optimization_log.append(
                f"✅ {model_name}: Best score = {search.best_score_:.4f} "
                f"(Improvement: {improvement:.4f})"
            )
        
        return optimized_models
    
    def _get_parameter_grid(self, model, model_name):
        """Get appropriate parameter grid for different model types."""
        model_type = type(model).__name__
        
        if 'RandomForest' in model_type:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        
        elif 'GradientBoosting' in model_type:
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        elif 'SVC' in model_type or 'SVR' in model_type:
            return {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        
        elif 'LogisticRegression' in model_type:
            return {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            }
        
        elif 'LinearRegression' in model_type:
            # Replace with Ridge for hyperparameter tuning
            from sklearn.linear_model import Ridge
            return {
                'alpha': [0.01, 0.1, 1, 10, 100]
            }
        
        else:
            return {}  # No parameters to tune
    
    def auto_optimize(self, models_dict, cv_folds=5, test_size=0.2):
        """
        Automatically detect and solve performance issues.
        """
        
        self.optimization_log.append("🤖 Starting Automatic Optimization...")
        
        # First, evaluate baseline performance
        baseline_scores = {}
        scoring = 'r2' if self.problem_type == 'regression' else 'accuracy'
        
        for model_name, model in models_dict.items():
            scores = cross_val_score(model, self.X, self.y, cv=cv_folds, scoring=scoring)
            baseline_scores[model_name] = np.mean(scores)
        
        if not baseline_scores:
            self.optimization_log.append("⚠️ No models to optimize.")
            return {}
            
        # Use the BEST model's score for the decision, not the average
        best_model_name = max(baseline_scores, key=baseline_scores.get)
        best_baseline_score = baseline_scores[best_model_name]
        
        self.optimization_log.append(f"📊 Best baseline performance ({best_model_name}): {best_baseline_score:.4f} ({scoring})")
        
        # Diagnose performance issues
        performance_issues = self.diagnose_performance_issues(models_dict, cv_folds)
        
        # Use the diagnostic results to choose the right strategy
        if performance_issues['severe_underfitting']:
            self.optimization_log.append("🚨 SEVERE UNDERFITTING detected. Applying aggressive solutions.")
            return self.solve_severe_underfitting(models_dict, cv_folds)
        
        # Prioritize fixing underfitting over overfitting
        elif performance_issues['underfitting']:
            self.optimization_log.append("⚠️ Moderate underfitting detected. Increasing model complexity.")
            return self.solve_underfitting(models_dict, cv_folds)
            
        elif performance_issues['overfitting']:
            self.optimization_log.append("🎯 High performance but potential overfitting detected. Applying regularization.")
            return self.solve_overfitting(models_dict, cv_folds, test_size)
            
        else:
            self.optimization_log.append("✅ All models seem to have reasonable performance. Applying standard hyperparameter tuning.")
            # Fallback to a standard optimization if no major issues are found
            return self.optimize_hyperparameters(models_dict, search_type='random', cv_folds=cv_folds)
    
    def solve_severe_underfitting(self, models_dict, cv_folds=5):
        """
        Solve severe underfitting with aggressive measures.
        """
        
        severe_solutions = {}
        
        self.optimization_log.append("🚨 Applying SEVERE underfitting solutions...")
        
        for model_name, model in models_dict.items():
            
            # Create much more complex models
            aggressive_model = self._create_aggressive_model(model, model_name)
            
            # Test the aggressive model
            scoring = 'r2' if self.problem_type == 'regression' else 'accuracy'
            scores = cross_val_score(aggressive_model, self.X, self.y, cv=cv_folds, scoring=scoring)
            new_score = np.mean(scores)
            
            # Get original score for comparison
            original_scores = cross_val_score(model, self.X, self.y, cv=cv_folds, scoring=scoring)
            original_score = np.mean(original_scores)
            
            severe_solutions[model_name] = {
                'best_model': aggressive_model,
                'approach': 'aggressive_complexity',
                'cv_score_mean': new_score,
                'improvement': new_score - original_score
            }
            
            self.optimization_log.append(
                f"✅ {model_name}: Aggressive model "
                f"(Score: {new_score:.4f} vs Original: {original_score:.4f}, "
                f"Improvement: {new_score - original_score:.4f})"
            )
        
        return severe_solutions
    
    def _create_aggressive_model(self, model, model_name):
        """Create extremely complex models to combat severe underfitting."""
        model_type = type(model).__name__
        
        if 'RandomForest' in model_type:
            # Maximum complexity Random Forest
            aggressive_model = type(model)(
                n_estimators=500,     # Many trees
                max_depth=None,       # No depth limit
                min_samples_split=2,  # Minimum splits
                min_samples_leaf=1,   # Minimum leaves
                max_features=None,    # Use all features
                bootstrap=True,
                random_state=42
            )
        
        elif 'GradientBoosting' in model_type:
            # High-complexity Gradient Boosting
            aggressive_model = type(model)(
                n_estimators=300,
                max_depth=8,          # Deep trees
                learning_rate=0.3,    # Higher learning rate
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        
        elif 'SVC' in model_type or 'SVR' in model_type:
            # Very flexible SVM
            aggressive_model = type(model)(
                kernel='rbf',
                C=1000,              # Very low regularization
                gamma='scale'
            )
        
        elif 'LogisticRegression' in model_type:
            # Remove all regularization
            from sklearn.linear_model import LogisticRegression
            aggressive_model = LogisticRegression(
                C=10000,             # Almost no regularization
                penalty=None,        # No penalty
                random_state=42,
                max_iter=2000
            )
        
        elif 'KNeighbors' in model_type:
            # Very local model
            aggressive_model = type(model)(n_neighbors=1)
        
        else:
            # Default: try to create a more complex version
            aggressive_model = model
        
        return aggressive_model
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        return {
            'optimization_log': self.optimization_log,
            'selected_features': self.selected_features,
            'n_selected_features': len(self.selected_features) if self.selected_features else 0,
            'best_hyperparameters': self.best_params,
            'problem_type': self.problem_type,
            'recommendations_implemented': [
                'Overfitting prevention through regularization and complexity reduction',
                'Feature selection to remove irrelevant features',
                'Hyperparameter optimization for best performance'
            ]
        }
