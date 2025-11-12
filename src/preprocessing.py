"""Preprocessing utilities for weather forecasting."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit


def to_dense(x):
    """Convert sparse matrix to dense array."""
    return x.toarray() if hasattr(x, 'toarray') else x


def build_preprocessor(numeric_cols, categorical_cols, text_cols):
    """
    Build sklearn ColumnTransformer for preprocessing.
    
    Parameters:
    -----------
    numeric_cols : list
        Names of numeric columns
    categorical_cols : list
        Names of categorical columns
    text_cols : list
        Names of text columns
        
    Returns:
    --------
    preprocessor : ColumnTransformer
        Fitted preprocessor
    """
    transformers = []
    
    if numeric_cols:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_cols))
    
    if categorical_cols:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_cols))
    
    if text_cols:
        text_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('flatten', FunctionTransformer(lambda x: x.ravel(), accept_sparse=False)),
            ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2)))
        ])
        transformers.append(('text', text_transformer, [text_cols[0]]))
    
    return ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.0)


def make_pipeline(estimator, numeric_cols, categorical_cols, text_cols):
    """
    Create complete preprocessing + model pipeline.
    
    Parameters:
    -----------
    estimator : sklearn estimator
        Model to use
    numeric_cols : list
        Names of numeric columns
    categorical_cols : list
        Names of categorical columns
    text_cols : list
        Names of text columns
        
    Returns:
    --------
    pipeline : Pipeline
        Complete pipeline
    """
    return Pipeline([
        ('preprocess', build_preprocessor(numeric_cols, categorical_cols, text_cols)),
        ('dense', FunctionTransformer(to_dense, accept_sparse=True)),
        ('regressor', estimator)
    ])


class ModTimeSeriesSplit:
    """
    Time series split that removes leakage buffer from validation sets.
    
    This prevents data leakage by ensuring validation set starts after
    a buffer period following the training set.
    """
    def __init__(self, n_splits=5, buffer=14):
        """
        Parameters:
        -----------
        n_splits : int
            Number of folds
        buffer : int
            Number of samples to skip between train and validation
        """
        self.n_splits = n_splits
        self.buffer = buffer
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def split(self, X, y=None, groups=None):
        """Generate train/validation indices with buffer."""
        for train_idx, val_idx in self.tscv.split(X):
            # Remove first 'buffer' samples from validation set
            val_idx_clean = val_idx[self.buffer:]
            if len(val_idx_clean) > 0:
                yield train_idx, val_idx_clean
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits


def identify_column_types(X_train, potential_categorical=None, potential_text=None):
    """
    Automatically identify column types for preprocessing.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    potential_categorical : list, optional
        Columns to consider as categorical
    potential_text : list, optional
        Columns to consider as text
        
    Returns:
    --------
    column_types : dict
        Dictionary with 'numeric', 'categorical', 'text' keys
    """
    if potential_categorical is None:
        potential_categorical = ['preciptype', 'conditions', 'icon', 'month', 'day_of_week', 'hour', 'year']
    
    if potential_text is None:
        potential_text = ['description']
    
    categorical_cols = [col for col in potential_categorical if col in X_train.columns]
    text_cols = [col for col in potential_text if col in X_train.columns]
    numeric_cols = [col for col in X_train.columns 
                   if col not in categorical_cols + text_cols]
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'text': text_cols
    }
