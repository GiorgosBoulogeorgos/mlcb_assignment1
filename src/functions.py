# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_dataset(filepath):
    """Load the dataset from CSV file."""
    return pd.read_csv(filepath)

def split_data(df, target_col='age', test_size=0.2, seed=42):
    """
    Split data into train/validation sets.
    Use binning to allow stratification on the continuous 'age' target.
    """
    # Create bins for stratification (Task 1.1)
    # We use 5 bins as a heuristic to ensure enough samples per bin
    age_bins = pd.cut(df[target_col], bins=5, labels=False)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=age_bins
    )
    return train_df, val_df

def get_preprocessing_pipeline(cpg_cols, categorical_cols):
    """
    Create a scikit-learn Pipeline for preprocessing.
    Task 1.2: Imputation (MCAR) and Scaling.
    """
    # Numeric pipeline: Impute missing values with mean (appropriate for MCAR)
    # and scale features.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Handle Sex and Ethnicity
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, cpg_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor

def get_feature_matrices(df, preprocessor, fit=False):
    """
    Applies the preprocessor to return a transformed feature matrix.
    If fit=True, it fits the preprocessor (use only for training set).
    """
    if fit:
        transformed_data = preprocessor.fit_transform(df)
    else:
        transformed_data = preprocessor.transform(df)

    # Get feature names after OneHotEncoding
    try:
        cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
        num_features = preprocessor.transformers_[0][2]
        all_features = list(num_features) + list(cat_features)
    except:
        all_features = None

    return transformed_data, all_features


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def calculate_metrics(y_true, y_pred):
    """Calculate the four required regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Pearson_r': r}

def bootstrap_eval(y_true, y_pred, n_resamples=1000, seed=42):
    """
    Perform bootstrap resampling on validation predictions.
    Task 2.1 requirement: 1000 resamples, 95% CI.
    """
    np.random.seed(seed)
    n_samples = len(y_true)
    bootstrap_results = []
    
    for _ in range(n_resamples):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        resample_metrics = calculate_metrics(y_true[indices], y_pred[indices])
        bootstrap_results.append(resample_metrics)
        
    boot_df = pd.DataFrame(bootstrap_results)
    
    # Calculate Mean and 95% CI (2.5th and 97.5th percentiles)
    summary = {}
    for col in boot_df.columns:
        summary[col] = {
            'mean': boot_df[col].mean(),
            'lo': boot_df[col].quantile(0.025),
            'hi': boot_df[col].quantile(0.975),
            'raw': boot_df[col].values
        }
    return summary



import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

def stability_selection(X, y, feature_names, n_subsamples=50, subsample_fraction=0.8, top_n=200, threshold=0.5, seed=42):
    """
    Implement stability selection by perturbing the training data.
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    subsample_size = int(n_samples * subsample_fraction)
    
    feature_counts = {name: 0 for name in feature_names}
    
    for _ in range(n_subsamples):
        # Draw subsample without replacement
        indices = np.random.choice(n_samples, size=subsample_size, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
        
        # Calculate absolute Spearman correlation
        correlations = []
        for i in range(X_sub.shape[1]):
            corr, _ = spearmanr(X_sub[:, i], y_sub)
            correlations.append(abs(corr))
            
        # Get indices of top_n features
        top_indices = np.argsort(correlations)[-top_n:]
        
        for idx in top_indices:
            feature_counts[feature_names[idx]] += 1
            
    # Keep features selected in > 50% of iterations
    stable_threshold = int(n_subsamples * threshold)
    stable_features = [name for name, count in feature_counts.items() if count > stable_threshold]
    
    return stable_features, feature_counts

def evaluate_proxy_model(X_train, y_train, X_val, y_val, feature_mask):
    """
    Evaluate a feature subset using ElasticNet (at defaults) as a proxy model.
    """
    model = ElasticNet(random_state=42)
    # Subset features
    X_tr_sub = X_train[:, feature_mask]
    X_val_sub = X_val[:, feature_mask]
    
    model.fit(X_tr_sub, y_train)
    preds = model.predict(X_val_sub)
    
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    
    return rmse, r2




from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from scipy.stats import loguniform, uniform

def tune_regression_models(X_train, y_train, n_iter=40, cv=5, seed=42):
    """
    Tunes ElasticNet, SVR, and Bayesian Ridge using RandomizedSearchCV.
    Objective is to minimize mean cross-validation RMSE (by using neg_root_mean_squared_error).
    """
    models_and_params = {
        'ElasticNet': {
            'model': ElasticNet(random_state=seed, max_iter=10000),
            'params': {
                'alpha': loguniform(0.001, 10),
                'l1_ratio': uniform(0.1, 0.9)  # 0.1 + 0.9 = 1.0 max
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': loguniform(0.1, 500),
                'epsilon': [0.01, 0.1, 0.5, 1.0],
                'kernel': ['rbf', 'linear']
            }
        },
        'Bayesian Ridge': {
            'model': BayesianRidge(),
            'params': {
                'alpha_1': loguniform(1e-7, 1e-3),
                'alpha_2': loguniform(1e-7, 1e-3),
                'lambda_1': loguniform(1e-7, 1e-3),
                'lambda_2': loguniform(1e-7, 1e-3)
            }
        }
    }
    
    best_estimators = {}
    
    for name, mp in models_and_params.items():
        print(f"Tuning {name}...")
        search = RandomizedSearchCV(
            estimator=mp['model'],
            param_distributions=mp['params'],
            n_iter=n_iter,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            random_state=seed,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        print(f"Best CV RMSE for {name}: {-search.best_score_:.4f}")
        best_estimators[name] = search.best_estimator_
        
    return best_estimators


import optuna
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
import numpy as np

def optuna_tune_model(model_name, pipeline, X_train, y_train, n_trials=40, cv=5, seed=42):
    """
    Tune models using Optuna (Tree-structured Parzen Estimator).
    Returns the best fitted pipeline/estimator and the Optuna study object.
    """
    def objective(trial):
        # Define hyperparameter search spaces based on the model
        if model_name == 'ElasticNet':
            alpha = trial.suggest_float('alpha', 0.001, 10.0, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.1, 1.0)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=10000)
        elif model_name == 'SVR':
            c = trial.suggest_float('C', 0.1, 500.0, log=True)
            epsilon = trial.suggest_categorical('epsilon', [0.01, 0.1, 0.5, 1.0])
            kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
            model = SVR(C=c, epsilon=epsilon, kernel=kernel)
        elif model_name == 'Bayesian Ridge':
            alpha_1 = trial.suggest_float('alpha_1', 1e-7, 1e-3, log=True)
            alpha_2 = trial.suggest_float('alpha_2', 1e-7, 1e-3, log=True)
            lambda_1 = trial.suggest_float('lambda_1', 1e-7, 1e-3, log=True)
            lambda_2 = trial.suggest_float('lambda_2', 1e-7, 1e-3, log=True)
            model = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Evaluate using cross-validation
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, 
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        return -scores.mean() # Return positive RMSE to minimize

    # Set up and run the Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    
    # Reconstruct and fit the best model
    best_params = study.best_params
    if model_name == 'ElasticNet':
        best_model = ElasticNet(**best_params, random_state=seed, max_iter=10000)
    elif model_name == 'SVR':
        best_model = SVR(**best_params)
    elif model_name == 'Bayesian Ridge':
        best_model = BayesianRidge(**best_params)
        
    best_model.fit(X_train, y_train)
    
    return best_model, study



import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

def calculate_class_metrics(y_true, y_pred, y_prob):
    """Calculate the required classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    return {'Accuracy': acc, 'F1': f1, 'MCC': mcc, 'ROC-AUC': roc_auc, 'PR-AUC': pr_auc}

def bootstrap_eval_class(y_true, y_pred, y_prob, n_resamples=1000, seed=42):
    """
    Perform bootstrap resampling on validation/evaluation predictions for classification.
    """
    np.random.seed(seed)
    n_samples = len(y_true)
    bootstrap_results = []
    
    for _ in range(n_resamples):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        # Ensure we have both classes in the resample to calculate AUC
        if len(np.unique(y_true[indices])) > 1:
            resample_metrics = calculate_class_metrics(y_true[indices], y_pred[indices], y_prob[indices])
            bootstrap_results.append(resample_metrics)
        
    boot_df = pd.DataFrame(bootstrap_results)
    
    summary = {}
    for col in boot_df.columns:
        summary[col] = {
            'mean': boot_df[col].mean(),
            'lo': boot_df[col].quantile(0.025),
            'hi': boot_df[col].quantile(0.975),
            'raw': boot_df[col].values
        }
    return summary