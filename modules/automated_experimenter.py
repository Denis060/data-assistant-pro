"""
Automated Model Experimenter
The final missing piece - automated experimentation with different configurations
Includes automated model selection, configuration tuning, and performance tracking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class AutomatedModelExperimenter:
    """
    The ultimate missing piece - automated experimentation across all possible configurations
    """
    
    def __init__(self, problem_type='auto'):
        self.problem_type = problem_type
        self.experiment_log = []
        self.experiment_results = {}
        self.best_configurations = {}
        
    def run_comprehensive_experiments(self, X, y, max_experiments=20, time_budget_minutes=30, include_ensembles=True, include_preprocessing=True):
        """
        Run comprehensive automated experiments to find the absolute best configuration
        """
        
        start_time = time.time()
        time_budget_seconds = time_budget_minutes * 60
        
        # Detect problem type if auto
        if self.problem_type == 'auto':
            self.problem_type = 'classification' if len(np.unique(y)) <= 10 else 'regression'
        
        self.experiment_log.append(f"🧪 Starting comprehensive experiments for {self.problem_type}")
        self.experiment_log.append(f"⏱️ Time budget: {time_budget_minutes} minutes, Max experiments: {max_experiments}")
        
        # Get all possible model configurations
        model_configs = self._get_all_model_configurations()
        
        experiment_count = 0
        
        for config_name, config in model_configs.items():
            
            # Check time budget
            elapsed_time = time.time() - start_time
            if elapsed_time > time_budget_seconds:
                self.experiment_log.append(f"⏰ Time budget exceeded ({elapsed_time:.1f}s), stopping experiments")
                break
            
            if experiment_count >= max_experiments:
                self.experiment_log.append(f"🔢 Max experiments reached ({max_experiments}), stopping")
                break
            
            self.experiment_log.append(f"🔬 Experiment {experiment_count + 1}: {config_name}")
            
            try:
                # Run experiment
                result = self._run_single_experiment(X, y, config, config_name)
                self.experiment_results[config_name] = result
                
                self.experiment_log.append(f"✅ {config_name}: Score = {result['cv_score']:.4f} (±{result['cv_std']:.4f})")
                
                experiment_count += 1
                
            except Exception as e:
                self.experiment_log.append(f"❌ {config_name} failed: {str(e)}")
                continue
        
        # Find best configuration
        self.best_configurations = self._analyze_experiment_results()
        
        total_time = time.time() - start_time
        self.experiment_log.append(f"🏁 Completed {experiment_count} experiments in {total_time:.1f} seconds")
        
        return {
            'experiment_results': self.experiment_results,
            'best_configurations': self.best_configurations,
            'experiment_log': self.experiment_log,
            'total_experiments': experiment_count,
            'total_time': total_time
        }
    
    def _get_all_model_configurations(self):
        """Get all possible model configurations to experiment with"""
        
        configs = {}
        
        if self.problem_type == 'classification':
            
            # Random Forest variations
            configs.update({
                'rf_small': {
                    'model': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
                    'description': 'Small Random Forest'
                },
                'rf_medium': {
                    'model': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                    'description': 'Medium Random Forest'
                },
                'rf_large': {
                    'model': RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42),
                    'description': 'Large Random Forest'
                },
                'rf_deep': {
                    'model': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42),
                    'description': 'Deep Random Forest'
                },
                
                # Gradient Boosting variations
                'gb_fast': {
                    'model': GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, max_depth=3, random_state=42),
                    'description': 'Fast Gradient Boosting'
                },
                'gb_balanced': {
                    'model': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
                    'description': 'Balanced Gradient Boosting'
                },
                'gb_accurate': {
                    'model': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42),
                    'description': 'Accurate Gradient Boosting'
                },
                
                # SVM variations
                'svm_linear': {
                    'model': SVC(kernel='linear', C=1.0, random_state=42),
                    'description': 'Linear SVM'
                },
                'svm_rbf': {
                    'model': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
                    'description': 'RBF SVM'
                },
                'svm_poly': {
                    'model': SVC(kernel='poly', degree=3, C=1.0, random_state=42),
                    'description': 'Polynomial SVM'
                },
                
                # Logistic Regression variations
                'lr_l1': {
                    'model': LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42),
                    'description': 'L1 Logistic Regression'
                },
                'lr_l2': {
                    'model': LogisticRegression(penalty='l2', C=1.0, random_state=42),
                    'description': 'L2 Logistic Regression'
                },
                'lr_elastic': {
                    'model': LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5, solver='saga', random_state=42),
                    'description': 'ElasticNet Logistic Regression'
                },
                
                # K-Nearest Neighbors variations
                'knn_small': {
                    'model': KNeighborsClassifier(n_neighbors=3),
                    'description': 'Small KNN (k=3)'
                },
                'knn_medium': {
                    'model': KNeighborsClassifier(n_neighbors=5),
                    'description': 'Medium KNN (k=5)'
                },
                'knn_large': {
                    'model': KNeighborsClassifier(n_neighbors=10),
                    'description': 'Large KNN (k=10)'
                },
                
                # Naive Bayes
                'nb_gaussian': {
                    'model': GaussianNB(),
                    'description': 'Gaussian Naive Bayes'
                },
                
                # Decision Tree variations
                'dt_shallow': {
                    'model': DecisionTreeClassifier(max_depth=5, random_state=42),
                    'description': 'Shallow Decision Tree'
                },
                'dt_deep': {
                    'model': DecisionTreeClassifier(max_depth=15, min_samples_split=10, random_state=42),
                    'description': 'Deep Decision Tree'
                }
            })
            
        else:  # Regression
            
            configs.update({
                # Random Forest variations
                'rf_small': {
                    'model': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
                    'description': 'Small Random Forest'
                },
                'rf_medium': {
                    'model': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                    'description': 'Medium Random Forest'
                },
                'rf_large': {
                    'model': RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42),
                    'description': 'Large Random Forest'
                },
                
                # Gradient Boosting variations
                'gb_fast': {
                    'model': GradientBoostingRegressor(n_estimators=50, learning_rate=0.2, max_depth=3, random_state=42),
                    'description': 'Fast Gradient Boosting'
                },
                'gb_balanced': {
                    'model': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
                    'description': 'Balanced Gradient Boosting'
                },
                'gb_accurate': {
                    'model': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42),
                    'description': 'Accurate Gradient Boosting'
                },
                
                # Linear models
                'linear': {
                    'model': LinearRegression(),
                    'description': 'Linear Regression'
                },
                'ridge': {
                    'model': Ridge(alpha=1.0, random_state=42),
                    'description': 'Ridge Regression'
                },
                'lasso': {
                    'model': Lasso(alpha=1.0, random_state=42),
                    'description': 'Lasso Regression'
                },
                'elastic': {
                    'model': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
                    'description': 'ElasticNet Regression'
                },
                
                # SVM variations
                'svm_linear': {
                    'model': SVR(kernel='linear', C=1.0),
                    'description': 'Linear SVR'
                },
                'svm_rbf': {
                    'model': SVR(kernel='rbf', C=1.0, gamma='scale'),
                    'description': 'RBF SVR'
                },
                
                # K-Nearest Neighbors variations
                'knn_small': {
                    'model': KNeighborsRegressor(n_neighbors=3),
                    'description': 'Small KNN (k=3)'
                },
                'knn_medium': {
                    'model': KNeighborsRegressor(n_neighbors=5),
                    'description': 'Medium KNN (k=5)'
                },
                'knn_large': {
                    'model': KNeighborsRegressor(n_neighbors=10),
                    'description': 'Large KNN (k=10)'
                },
                
                # Decision Tree variations
                'dt_shallow': {
                    'model': DecisionTreeRegressor(max_depth=5, random_state=42),
                    'description': 'Shallow Decision Tree'
                },
                'dt_deep': {
                    'model': DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42),
                    'description': 'Deep Decision Tree'
                }
            })
        
        return configs
    
    def _run_single_experiment(self, X, y, config, config_name):
        """Run a single experiment with the given configuration"""
        
        model = config['model']
        description = config['description']
        
        start_time = time.time()
        
        # Cross-validation
        scoring = 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        
        training_time = time.time() - start_time
        
        # Additional metrics for detailed analysis
        model.fit(X, y)
        y_pred = model.predict(X)
        
        if self.problem_type == 'classification':
            additional_metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
            }
        else:
            additional_metrics = {
                'mse': mean_squared_error(y, y_pred),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            }
        
        return {
            'config_name': config_name,
            'description': description,
            'model': model,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist(),
            'training_time': training_time,
            'additional_metrics': additional_metrics
        }
    
    def _analyze_experiment_results(self):
        """Analyze all experiment results to find the best configurations"""
        
        if not self.experiment_results:
            return {}
        
        # Sort by CV score
        sorted_results = sorted(
            self.experiment_results.items(),
            key=lambda x: x[1]['cv_score'],
            reverse=True
        )
        
        best_overall = sorted_results[0][1]
        
        # Find best in each category
        categories = {
            'fast': [],      # Models with training time < 1 second
            'accurate': [],  # Top 3 performing models
            'stable': [],    # Models with low variance (cv_std < 0.02)
            'balanced': []   # Good balance of speed and accuracy
        }
        
        for name, result in self.experiment_results.items():
            
            # Fast models
            if result['training_time'] < 1.0:
                categories['fast'].append((name, result))
            
            # Stable models (low variance)
            if result['cv_std'] < 0.02:
                categories['stable'].append((name, result))
            
            # Balanced models (good score + reasonable time)
            if result['cv_score'] > (best_overall['cv_score'] * 0.95) and result['training_time'] < 5.0:
                categories['balanced'].append((name, result))
        
        # Top 3 accurate
        categories['accurate'] = sorted_results[:3]
        
        # Get best from each category
        best_configs = {}
        
        for category, results in categories.items():
            if results:
                if category == 'fast':
                    best_configs[category] = max(results, key=lambda x: x[1]['cv_score'])
                elif category == 'stable':
                    best_configs[category] = max(results, key=lambda x: x[1]['cv_score'])
                elif category == 'balanced':
                    # Score by balanced metric (score / log(time + 1))
                    best_configs[category] = max(results, key=lambda x: x[1]['cv_score'] / np.log(x[1]['training_time'] + 1))
                else:  # accurate
                    best_configs[category] = results[0]  # Already sorted by score
        
        return {
            'overall_best': best_overall,
            'category_best': best_configs,
            'all_results_ranked': sorted_results
        }
    
    def get_recommendations(self):
        """Get final recommendations based on all experiments"""
        
        if not self.best_configurations:
            return "No experiments completed yet. Run comprehensive experiments first."
        
        recommendations = []
        
        # Overall best
        overall_best = self.best_configurations['overall_best']
        recommendations.append(f"🏆 **Overall Best**: {overall_best['config_name']} - {overall_best['description']}")
        recommendations.append(f"   Score: {overall_best['cv_score']:.4f} (±{overall_best['cv_std']:.4f})")
        
        # Category recommendations
        for category, (name, result) in self.best_configurations['category_best'].items():
            if category == 'fast':
                recommendations.append(f"⚡ **Fastest**: {name} - {result['description']} ({result['training_time']:.2f}s)")
            elif category == 'stable':
                recommendations.append(f"🛡️ **Most Stable**: {name} - {result['description']} (σ={result['cv_std']:.4f})")
            elif category == 'balanced':
                recommendations.append(f"⚖️ **Best Balanced**: {name} - {result['description']}")
            elif category == 'accurate':
                recommendations.append(f"🎯 **Most Accurate**: {name} - {result['description']}")
        
        return recommendations
    
    def export_experiment_results(self):
        """Export all experiment results for analysis"""
        
        export_data = []
        
        for name, result in self.experiment_results.items():
            row = {
                'config_name': name,
                'description': result['description'],
                'cv_score': result['cv_score'],
                'cv_std': result['cv_std'],
                'training_time': result['training_time']
            }
            
            # Add additional metrics
            for metric, value in result['additional_metrics'].items():
                row[metric] = value
            
            export_data.append(row)
        
        return pd.DataFrame(export_data).sort_values('cv_score', ascending=False)
