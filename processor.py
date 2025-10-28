
"""
Data Preprocessor for ML
Load CSV (via pandas), convert to NumPy, normalize columns, handle NaN values.
Create feature matrix X and label vector y.

"""

import pandas as pd
import numpy as np
from typing import Tuple

class MLDataPreprocessor:
    """
    A comprehensive data preprocessor for machine learning workflows.
    Handles CSV loading, conversion to NumPy, normalization, and NaN handling.
    """
    
    def __init__(self):
        self.feature_means = None
        self.feature_stds = None
        self.column_names = None
        
    def load_csv(self, filepath='D:\data_preprocessor _for_ml\sales_data_sample.csv' ) -> pd.DataFrame:
        """
        Load CSV file using pandas.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            pandas DataFrame
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    
    def handle_missing_values(self, data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
        """
        Handle NaN values in the dataset.
        
        Args:
            data: NumPy array with potential NaN values
            strategy: 'mean', 'median', 'zero', or 'drop'
            
        Returns:
            Cleaned NumPy array
        """
        print(f"\nHandling missing values using '{strategy}' strategy...")
        nan_count = np.sum(np.isnan(data))
        print(f"Found {nan_count} NaN values")
        
        if strategy == 'mean':
            # Replace NaN with column mean
            col_mean = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = np.take(col_mean, inds[1])
            
        elif strategy == 'median':
            # Replace NaN with column median
            col_median = np.nanmedian(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = np.take(col_median, inds[1])
            
        elif strategy == 'zero':
            # Replace NaN with 0
            data = np.nan_to_num(data, nan=0.0)
            
        elif strategy == 'drop':
            # Remove rows with NaN
            data = data[~np.isnan(data).any(axis=1)]
            print(f"Dropped rows, new shape: {data.shape}")
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"Missing values handled")
        return data
    
    def normalize_features(self, X: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        Normalize features using standardization or min-max scaling.
        
        Args:
            X: Feature matrix
            method: 'standard' (z-score) or 'minmax'
            
        Returns:
            Normalized feature matrix
        """
        print(f"\nNormalizing features using '{method}' method...")
        
        if method == 'standard':
            # Z-score normalization: (X - mean) / std
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            
            # Avoid division by zero
            self.feature_stds[self.feature_stds == 0] = 1
            
            X_normalized = (X - self.feature_means) / self.feature_stds
            
        elif method == 'minmax':
            # Min-Max scaling: (X - min) / (max - min)
            X_min = np.min(X, axis=0)
            X_max = np.max(X, axis=0)
            
            # Avoid division by zero
            X_range = X_max - X_min
            X_range[X_range == 0] = 1
            
            X_normalized = (X - X_min) / X_range
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Features normalized")
        return X_normalized
    
    def split_features_labels(self, df: pd.DataFrame, label_column:str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split DataFrame into feature matrix X and label vector y.
        
        Args:
            df: pandas DataFrame
            label_column: Name of the target/label column
            
        Returns:
            Tuple of (X, y) as NumPy arrays
        """
        print(f"\nSplitting features and labels...")
        print(f"Label column: '{label_column}'")
        
        # Extract labels
        y = df[label_column].values
        
        # Extract features (all columns except label)
        X = df.drop(columns=[label_column]).values
        
        # Store column names for reference
        self.column_names = df.drop(columns=[label_column]).columns.tolist()
        
        print(f"✓ Feature matrix X: {X.shape}")
        print(f"✓ Label vector y: {y.shape}")
        
        return X, y
    
    def preprocess_pipeline(self, 
                           filepath: str, 
                           label_column: str,
                           nan_strategy: str = 'mean',
                           normalize_method: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline: Load → Split → Handle NaN → Normalize.
        
        Args:
            filepath: Path to CSV file
            label_column: Name of target column
            nan_strategy: Strategy for handling NaN values
            normalize_method: Normalization method
            
        Returns:
            Preprocessed (X, y) tuple
        """
        print("=" * 60)
        print("STARTING ML DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load CSV
        df = self.load_csv(filepath)
        
        # Step 2: Split features and labels
        X, y = self.split_features_labels(df, label_column)
        
        # Step 3: Handle missing values in features
        X = self.handle_missing_values(X, strategy=nan_strategy)
        
        # Step 4: Normalize features
        X = self.normalize_features(X, method=normalize_method)
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Final X shape: {X.shape}")
        print(f"Final y shape: {y.shape}")
        print(f"\nFeature statistics:")
        print(f"  Mean: {np.mean(X, axis=0)[:5]}..." if X.shape[1] > 5 else f"  Mean: {np.mean(X, axis=0)}")
        print(f"  Std:  {np.std(X, axis=0)[:5]}..." if X.shape[1] > 5 else f"  Std:  {np.std(X, axis=0)}")
        
        return X, y
    
    def get_data_summary(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Generate summary statistics about the preprocessed data.
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_names': self.column_names,
            'X_mean': np.mean(X, axis=0),
            'X_std': np.std(X, axis=0),
            'X_min': np.min(X, axis=0),
            'X_max': np.max(X, axis=0),
            'y_unique_values': np.unique(y),
            'y_distribution': {val: np.sum(y == val) for val in np.unique(y)}
        }
        return summary


# EXAMPLE USAGE AND TESTING
if __name__ == "__main__":
    # Create sample data for demonstration
    print("Creating sample dataset for demonstration...\n")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    
    sample_data = {
        'feature_1': np.random.randn(n_samples) * 10 + 50,
        'feature_2': np.random.randn(n_samples) * 5 + 20,
        'feature_3': np.random.randn(n_samples) * 15 + 100,
        'feature_4': np.random.randn(n_samples) * 3 + 7,
        'label': np.random.randint(0, 2, n_samples)
    }
    
    # Add some NaN values
    sample_data['feature_1'][10:15] = np.nan
    sample_data['feature_3'][20:25] = np.nan
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(sample_data)
    df.to_csv('new_data.csv', index=False)
    
    # Initialize preprocessor
    preprocessor = MLDataPreprocessor()
    
    # Run the complete pipeline
    X, y = preprocessor.preprocess_pipeline(
        filepath='new_data.csv',
        label_column='label',
        nan_strategy='mean',
        normalize_method='standard'
    )
    
    # Get and display summary
    summary = preprocessor.get_data_summary(X, y)
    print(f"\n....DATA SUMMARY....")
    print(f"Samples: {summary['n_samples']}")
    print(f"Features: {summary['n_features']}")
    print(f"Label distribution: {summary['y_distribution']}")
    
    print("\n Data is ready for machine learning!")
    