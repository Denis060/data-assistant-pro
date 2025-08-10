"""
Automated ML Pipeline
The missing pieces for complete ML performance optimization
Includes ensemble methods, automated feature engineering, data augmentation, and model stacking
"""

import os
# Fix scipy/sklearn compatibility issue
os.environ['SCIPY_ARRAY_API'] = '1'

import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.stats import boxcox, yeojohnson

# Try to import imbalanced-learn with fallback
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import EditedNearestNeighbours
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("⚠️ imbalanced-learn not available, using fallback augmentation methods")

import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class AutomatedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Automated feature engineering that creates powerful new features
    """
    
    def __init__(self, polynomial_degree=2, interaction_only=False, 
                 create_ratios=True, create_binning=True, create_datetime_features=True):
        self.polynomial_degree = polynomial_degree
        self.interaction_only = interaction_only
        self.create_ratios = create_ratios
        self.create_binning = create_binning
        self.create_datetime_features = create_datetime_features
        self.feature_names = None
        self.numeric_features = None
        self.poly_features = None
        
    def fit(self, X, y=None):
        """Learn feature patterns from training data"""
        self.feature_names = X.columns.tolist()
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fit polynomial features
        if len(self.numeric_features) > 0:
            self.poly_features = PolynomialFeatures(
                degree=self.polynomial_degree, 
                interaction_only=self.interaction_only,
                include_bias=False
            )
            self.poly_features.fit(X[self.numeric_features])
        
        return self
    
    def transform(self, X):
        """Create new engineered features"""
        X_new = X.copy()
        
        # 1. Polynomial and Interaction Features
        if self.poly_features and len(self.numeric_features) > 0:
            poly_array = self.poly_features.transform(X[self.numeric_features])
            poly_names = self.poly_features.get_feature_names_out(self.numeric_features)
            
            # Add only the new features (not the original ones)
            new_poly_features = []
            for i, name in enumerate(poly_names):
                if name not in self.numeric_features:  # Skip original features
                    X_new[f'poly_{name}'] = poly_array[:, i]
                    new_poly_features.append(f'poly_{name}')
        
        # 2. Ratio Features (very powerful for ML)
        if self.create_ratios and len(self.numeric_features) >= 2:
            for i, feat1 in enumerate(self.numeric_features):
                for feat2 in self.numeric_features[i+1:]:
                    # Avoid division by zero
                    denominator = X_new[feat2].replace(0, 1e-8)
                    X_new[f'ratio_{feat1}_{feat2}'] = X_new[feat1] / denominator
        
        # 3. Statistical Features (rolling statistics, percentiles)
        if len(self.numeric_features) > 0:
            for feat in self.numeric_features:
                # Percentile-based features
                q25, q75 = X_new[feat].quantile([0.25, 0.75])
                X_new[f'{feat}_q25_distance'] = abs(X_new[feat] - q25)
                X_new[f'{feat}_q75_distance'] = abs(X_new[feat] - q75)
                X_new[f'{feat}_iqr_position'] = (X_new[feat] - q25) / (q75 - q25 + 1e-8)
        
        # 4. Binning Features (convert continuous to categorical insights)
        if self.create_binning:
            for feat in self.numeric_features:
                # Equal-width binning
                X_new[f'{feat}_bin_equal'] = pd.cut(X_new[feat], bins=5, labels=False, duplicates='drop')
                
                # Quantile-based binning
                X_new[f'{feat}_bin_quantile'] = pd.qcut(X_new[feat], q=5, labels=False, duplicates='drop')
        
        # 5. Aggregation Features (if we have groups)
        if len(self.numeric_features) >= 3:
            # Create aggregate features
            X_new['feature_sum'] = X_new[self.numeric_features].sum(axis=1)
            X_new['feature_mean'] = X_new[self.numeric_features].mean(axis=1)
            X_new['feature_std'] = X_new[self.numeric_features].std(axis=1)
            X_new['feature_max'] = X_new[self.numeric_features].max(axis=1)
            X_new['feature_min'] = X_new[self.numeric_features].min(axis=1)
            X_new['feature_range'] = X_new['feature_max'] - X_new['feature_min']
        
        return X_new

class AdvancedDataAugmenter:
    """
    Advanced data augmentation techniques for improving model performance
    """
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.augmentation_log = []
        
    def augment_data(self, X, y, augmentation_strategy='auto'):
        """
        Apply advanced data augmentation techniques
        """
        
        if augmentation_strategy == 'auto':
            # Auto-select best augmentation based on data characteristics
            augmentation_strategy = self._select_augmentation_strategy(X, y)
        
        self.augmentation_log.append(f"🔄 Applying {augmentation_strategy} augmentation")
        
        if augmentation_strategy == 'smote':
            return self._apply_smote(X, y)
        elif augmentation_strategy == 'adasyn':
            return self._apply_adasyn(X, y)
        elif augmentation_strategy == 'smoteenn':
            return self._apply_smoteenn(X, y)
        elif augmentation_strategy == 'noise_injection':
            return self._apply_noise_injection(X, y)
        elif augmentation_strategy == 'bootstrap':
            return self._apply_bootstrap(X, y)
        else:
            return X, y
    
    def _select_augmentation_strategy(self, X, y):
        """Intelligently select the best augmentation strategy"""
        
        if self.problem_type == 'classification':
            # Check class imbalance
            class_counts = pd.Series(y).value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            
            if imbalance_ratio > 3:
                return 'smoteenn'  # Best for imbalanced data
            elif len(np.unique(y)) > 5:
                return 'adasyn'  # Good for multi-class
            else:
                return 'smote'  # Standard choice
        else:
            # Regression - use noise injection or bootstrap
            if len(X) < 1000:
                return 'bootstrap'  # Good for small datasets
            else:
                return 'noise_injection'  # Good for larger datasets
    
    def _apply_smote(self, X, y):
        """Apply SMOTE for synthetic minority oversampling"""
        if not IMBLEARN_AVAILABLE:
            self.augmentation_log.append("⚠️ SMOTE not available, using bootstrap instead")
            return self._apply_bootstrap(X, y)
            
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(np.unique(y)) - 1))
            X_resampled, y_resampled = smote.fit_resample(X, y)
            self.augmentation_log.append(f"✅ SMOTE: {len(X)} → {len(X_resampled)} samples")
            return X_resampled, y_resampled
        except:
            self.augmentation_log.append("⚠️ SMOTE failed, returning original data")
            return X, y
    
    def _apply_adasyn(self, X, y):
        """Apply ADASYN for adaptive synthetic sampling"""
        if not IMBLEARN_AVAILABLE:
            self.augmentation_log.append("⚠️ ADASYN not available, using bootstrap instead")
            return self._apply_bootstrap(X, y)
            
        try:
            adasyn = ADASYN(random_state=42, n_neighbors=min(5, len(np.unique(y)) - 1))
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            self.augmentation_log.append(f"✅ ADASYN: {len(X)} → {len(X_resampled)} samples")
            return X_resampled, y_resampled
        except:
            self.augmentation_log.append("⚠️ ADASYN failed, returning original data")
            return X, y
    
    def _apply_smoteenn(self, X, y):
        """Apply SMOTE + Edited Nearest Neighbours for combined over/under sampling"""
        if not IMBLEARN_AVAILABLE:
            self.augmentation_log.append("⚠️ SMOTEENN not available, using bootstrap instead")
            return self._apply_bootstrap(X, y)
            
        try:
            smoteenn = SMOTEENN(random_state=42)
            X_resampled, y_resampled = smoteenn.fit_resample(X, y)
            self.augmentation_log.append(f"✅ SMOTEENN: {len(X)} → {len(X_resampled)} samples")
            return X_resampled, y_resampled
        except:
            self.augmentation_log.append("⚠️ SMOTEENN failed, returning original data")
            return X, y
    
    def _apply_noise_injection(self, X, y):
        """Add Gaussian noise for data augmentation (good for regression)"""
        try:
            noise_factor = 0.05  # 5% noise
            X_noise = X + np.random.normal(0, noise_factor * X.std(), X.shape)
            
            # Combine original and noisy data
            X_augmented = np.vstack([X, X_noise])
            y_augmented = np.hstack([y, y])
            
            self.augmentation_log.append(f"✅ Noise Injection: {len(X)} → {len(X_augmented)} samples")
            return X_augmented, y_augmented
        except:
            self.augmentation_log.append("⚠️ Noise injection failed, returning original data")
            return X, y
    
    def _apply_bootstrap(self, X, y):
        """Bootstrap sampling for data augmentation"""
        try:
            n_bootstrap = min(len(X), 1000)  # Don't create too many samples
            bootstrap_indices = np.random.choice(len(X), size=n_bootstrap, replace=True)
            
            X_bootstrap = X.iloc[bootstrap_indices] if isinstance(X, pd.DataFrame) else X[bootstrap_indices]
            y_bootstrap = y.iloc[bootstrap_indices] if isinstance(y, pd.Series) else y[bootstrap_indices]
            
            # Combine original and bootstrap data
            if isinstance(X, pd.DataFrame):
                X_augmented = pd.concat([X, X_bootstrap], ignore_index=True)
                y_augmented = pd.concat([y, y_bootstrap], ignore_index=True)
            else:
                X_augmented = np.vstack([X, X_bootstrap])
                y_augmented = np.hstack([y, y_bootstrap])
            
            self.augmentation_log.append(f"✅ Bootstrap: {len(X)} → {len(X_augmented)} samples")
            return X_augmented, y_augmented
        except:
            self.augmentation_log.append("⚠️ Bootstrap failed, returning original data")
            return X, y

class EnsembleModelBuilder:
    """
    Advanced ensemble methods for superior performance
    """
    
    def __init__(self, problem_type='classification'):
        self.problem_type = problem_type
        self.ensemble_models = {}
        self.ensemble_log = []
        
    def create_voting_ensemble(self, models_dict):
        """Create voting ensemble from multiple models"""
        
        estimators = [(name, model) for name, model in models_dict.items()]
        
        if self.problem_type == 'classification':
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft'  # Use probability-based voting
            )
        else:
            ensemble = VotingRegressor(estimators=estimators)
        
        self.ensemble_models['voting'] = ensemble
        self.ensemble_log.append(f"✅ Created voting ensemble with {len(estimators)} models")
        
        return ensemble
    
    def create_stacking_ensemble(self, models_dict, meta_learner=None):
        """Create stacking ensemble with meta-learner"""
        
        estimators = [(name, model) for name, model in models_dict.items()]
        
        if meta_learner is None:
            if self.problem_type == 'classification':
                from sklearn.linear_model import LogisticRegression
                meta_learner = LogisticRegression(random_state=42)
            else:
                from sklearn.linear_model import Ridge
                meta_learner = Ridge(random_state=42)
        
        if self.problem_type == 'classification':
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5
            )
        else:
            ensemble = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5
            )
        
        self.ensemble_models['stacking'] = ensemble
        self.ensemble_log.append(f"✅ Created stacking ensemble with {len(estimators)} base models")
        
        return ensemble
    
    def create_bagging_ensemble(self, base_model):
        """Create bagging ensemble for variance reduction"""
        
        if self.problem_type == 'classification':
            ensemble = BaggingClassifier(
                estimator=base_model,
                n_estimators=10,
                random_state=42
            )
        else:
            ensemble = BaggingRegressor(
                estimator=base_model,
                n_estimators=10,
                random_state=42
            )
        
        self.ensemble_models['bagging'] = ensemble
        self.ensemble_log.append("✅ Created bagging ensemble")
        
        return ensemble
    
    def create_boosting_ensemble(self, base_model):
        """Create boosting ensemble for bias reduction"""
        
        if self.problem_type == 'classification':
            ensemble = AdaBoostClassifier(
                estimator=base_model,
                n_estimators=50,
                random_state=42
            )
        else:
            ensemble = AdaBoostRegressor(
                estimator=base_model,
                n_estimators=50,
                random_state=42
            )
        
        self.ensemble_models['boosting'] = ensemble
        self.ensemble_log.append("✅ Created boosting ensemble")
        
        return ensemble
    
    def get_all_ensembles(self):
        """Return all created ensemble models"""
        return self.ensemble_models

class AutomatedMLPipeline:
    """
    Complete automated ML pipeline with all missing pieces
    """
    
    def __init__(self, problem_type='auto'):
        self.problem_type = problem_type
        self.pipeline_log = []
        self.feature_engineer = None
        self.data_augmenter = None
        self.ensemble_builder = None
        self.pipelines = {}
        
    def create_complete_pipeline(self, X, y, base_models_dict, 
                                enable_feature_engineering=True,
                                enable_data_augmentation=True,
                                enable_ensembles=True,
                                enable_advanced_scaling=True):
        """
        Create the complete automated ML pipeline with all optimizations
        """
        
        # Detect problem type if auto
        if self.problem_type == 'auto':
            self.problem_type = 'classification' if len(np.unique(y)) <= 10 else 'regression'
        
        self.pipeline_log.append(f"🚀 Starting complete ML pipeline for {self.problem_type}")
        
        results = {}
        
        # 1. Advanced Feature Engineering
        if enable_feature_engineering:
            self.pipeline_log.append("🔧 Phase 1: Advanced Feature Engineering")
            self.feature_engineer = AutomatedFeatureEngineer(
                polynomial_degree=2,
                interaction_only=False,
                create_ratios=True,
                create_binning=True
            )
            
            X_engineered = self.feature_engineer.fit_transform(X)
            self.pipeline_log.append(f"✅ Features: {X.shape[1]} → {X_engineered.shape[1]}")
            results['feature_engineering'] = {
                'original_features': X.shape[1],
                'engineered_features': X_engineered.shape[1],
                'improvement': X_engineered.shape[1] - X.shape[1]
            }
        else:
            X_engineered = X
        
        # 2. Advanced Data Augmentation
        if enable_data_augmentation:
            self.pipeline_log.append("🔄 Phase 2: Advanced Data Augmentation")
            self.data_augmenter = AdvancedDataAugmenter(self.problem_type)
            X_augmented, y_augmented = self.data_augmenter.augment_data(X_engineered, y)
            
            results['data_augmentation'] = {
                'original_samples': len(X),
                'augmented_samples': len(X_augmented),
                'improvement': len(X_augmented) - len(X),
                'augmentation_log': self.data_augmenter.augmentation_log
            }
        else:
            X_augmented, y_augmented = X_engineered, y
        
        # 3. Advanced Scaling and Preprocessing
        if enable_advanced_scaling:
            self.pipeline_log.append("📊 Phase 3: Advanced Scaling")
            # Use RobustScaler instead of StandardScaler for better outlier handling
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X_augmented),
                columns=X_augmented.columns if hasattr(X_augmented, 'columns') else range(X_augmented.shape[1])
            )
            results['scaling'] = {'method': 'RobustScaler', 'status': 'completed'}
        else:
            X_scaled = X_augmented
        
        # 4. Train Enhanced Base Models
        self.pipeline_log.append("🤖 Phase 4: Training Enhanced Base Models")
        enhanced_models = {}
        for name, model in base_models_dict.items():
            try:
                enhanced_model = model.fit(X_scaled, y_augmented)
                enhanced_models[name] = enhanced_model
                self.pipeline_log.append(f"✅ Trained enhanced {name}")
            except Exception as e:
                self.pipeline_log.append(f"❌ Failed to train {name}: {str(e)}")
        
        # 5. Create Ensemble Models
        if enable_ensembles and len(enhanced_models) >= 2:
            self.pipeline_log.append("🎭 Phase 5: Creating Ensemble Models")
            self.ensemble_builder = EnsembleModelBuilder(self.problem_type)
            
            # Create different types of ensembles
            voting_ensemble = self.ensemble_builder.create_voting_ensemble(enhanced_models)
            stacking_ensemble = self.ensemble_builder.create_stacking_ensemble(enhanced_models)
            
            # Train ensembles
            voting_ensemble.fit(X_scaled, y_augmented)
            stacking_ensemble.fit(X_scaled, y_augmented)
            
            enhanced_models['voting_ensemble'] = voting_ensemble
            enhanced_models['stacking_ensemble'] = stacking_ensemble
            
            results['ensembles'] = {
                'voting_created': True,
                'stacking_created': True,
                'ensemble_log': self.ensemble_builder.ensemble_log
            }
        
        # 6. Model Evaluation with Cross-Validation
        self.pipeline_log.append("📊 Phase 6: Advanced Model Evaluation")
        evaluation_results = {}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if self.problem_type == 'classification' else KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
        
        for name, model in enhanced_models.items():
            try:
                cv_scores = cross_val_score(model, X_scaled, y_augmented, cv=cv, scoring=scoring)
                evaluation_results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                }
                self.pipeline_log.append(f"✅ {name}: CV Score = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            except Exception as e:
                self.pipeline_log.append(f"❌ Failed to evaluate {name}: {str(e)}")
        
        results['evaluation'] = evaluation_results
        results['final_models'] = enhanced_models
        results['pipeline_log'] = self.pipeline_log
        
        return results
    
    def get_best_model(self, results):
        """Get the best performing model from the pipeline results"""
        if 'evaluation' not in results:
            return None
        
        evaluation_results = results['evaluation']
        
        # Find best model based on CV score
        best_model_name = max(evaluation_results.keys(), key=lambda x: evaluation_results[x]['cv_mean'])
        best_model = results['final_models'][best_model_name]
        
        return {
            'name': best_model_name,
            'model': best_model,
            'cv_score': evaluation_results[best_model_name]['cv_mean'],
            'cv_std': evaluation_results[best_model_name]['cv_std']
        }
    
    def generate_pipeline_report(self, results):
        """Generate comprehensive pipeline performance report"""
        
        report = {
            'pipeline_phases': len([log for log in self.pipeline_log if 'Phase' in log]),
            'total_improvements': {},
            'best_model': self.get_best_model(results),
            'pipeline_log': self.pipeline_log
        }
        
        # Calculate total improvements
        if 'feature_engineering' in results:
            report['total_improvements']['feature_engineering'] = results['feature_engineering']['improvement']
        
        if 'data_augmentation' in results:
            report['total_improvements']['data_augmentation'] = results['data_augmentation']['improvement']
        
        if 'ensembles' in results:
            report['total_improvements']['ensemble_methods'] = len([k for k in results['final_models'].keys() if 'ensemble' in k])
        
        return report
