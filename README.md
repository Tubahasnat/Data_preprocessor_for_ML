# ML Data Preprocessor

A comprehensive Python library for preprocessing machine learning datasets. This tool streamlines the data preparation workflow by handling CSV loading, missing value imputation, feature normalization, and train-ready data formatting.

## Features

- **CSV Data Loading**: Easy loading of datasets using pandas
- **Missing Value Handling**: Multiple strategies for handling NaN values
  - Mean imputation
  - Median imputation
  - Zero filling
  - Row dropping
- **Feature Normalization**: Two normalization methods
  - Standard scaling (z-score normalization)
  - Min-Max scaling
- **Data Splitting**: Automatic separation of features and labels
- **Pipeline Processing**: Complete end-to-end preprocessing pipeline
- **Data Summary**: Comprehensive statistics about your processed data

## Installation

### Requirements

```bash
pip install pandas numpy
```

## Quick Start

```python
from data_preprocessor import MLDataPreprocessor

# Initialize the preprocessor
preprocessor = MLDataPreprocessor()

# Run the complete preprocessing pipeline
X, y = preprocessor.preprocess_pipeline(
    filepath='your_data.csv',
    label_column='target',
    nan_strategy='mean',
    normalize_method='standard'
)

# Your data is now ready for machine learning!
```

## Usage Examples

### Basic Usage

```python
import numpy as np
import pandas as pd
from data_preprocessor import MLDataPreprocessor

# Create preprocessor instance
preprocessor = MLDataPreprocessor()

# Load your CSV file
df = preprocessor.load_csv('data.csv')

# Split features and labels
X, y = preprocessor.split_features_labels(df, label_column='target')

# Handle missing values
X = preprocessor.handle_missing_values(X, strategy='mean')

# Normalize features
X = preprocessor.normalize_features(X, method='standard')
```

### Using the Complete Pipeline

```python
preprocessor = MLDataPreprocessor()

# One-line preprocessing
X, y = preprocessor.preprocess_pipeline(
    filepath='sales_data.csv',
    label_column='sales',
    nan_strategy='median',
    normalize_method='minmax'
)

# Get data summary
summary = preprocessor.get_data_summary(X, y)
print(f"Samples: {summary['n_samples']}")
print(f"Features: {summary['n_features']}")
print(f"Label distribution: {summary['y_distribution']}")
```

### Testing with Sample Data

The repository includes a test script that generates synthetic data:

```python
python data_preprocessor.py
```

This will:
1. Generate a sample dataset with 200 samples and 4 features
2. Introduce some NaN values
3. Run the preprocessing pipeline
4. Display summary statistics

## API Reference

### MLDataPreprocessor

#### Methods

**`load_csv(filepath)`**
- Loads a CSV file into a pandas DataFrame
- **Parameters**: `filepath` (str) - Path to CSV file
- **Returns**: pandas DataFrame

**`handle_missing_values(data, strategy='mean')`**
- Handles NaN values in the dataset
- **Parameters**: 
  - `data` (np.ndarray) - NumPy array with potential NaN values
  - `strategy` (str) - One of: 'mean', 'median', 'zero', 'drop'
- **Returns**: Cleaned NumPy array

**`normalize_features(X, method='standard')`**
- Normalizes feature matrix
- **Parameters**:
  - `X` (np.ndarray) - Feature matrix
  - `method` (str) - One of: 'standard' (z-score), 'minmax'
- **Returns**: Normalized feature matrix

**`split_features_labels(df, label_column)`**
- Splits DataFrame into features and labels
- **Parameters**:
  - `df` (pd.DataFrame) - Input DataFrame
  - `label_column` (str) - Name of target column
- **Returns**: Tuple (X, y) as NumPy arrays

**`preprocess_pipeline(filepath, label_column, nan_strategy='mean', normalize_method='standard')`**
- Complete preprocessing pipeline
- **Parameters**:
  - `filepath` (str) - Path to CSV file
  - `label_column` (str) - Name of target column
  - `nan_strategy` (str) - Strategy for handling NaN values
  - `normalize_method` (str) - Normalization method
- **Returns**: Preprocessed (X, y) tuple

**`get_data_summary(X, y)`**
- Generates summary statistics
- **Parameters**:
  - `X` (np.ndarray) - Feature matrix
  - `y` (np.ndarray) - Label vector
- **Returns**: Dictionary with summary statistics

## Configuration Options

### Missing Value Strategies
- `'mean'`: Replace NaN with column mean
- `'median'`: Replace NaN with column median
- `'zero'`: Replace NaN with 0
- `'drop'`: Remove rows containing NaN

### Normalization Methods
- `'standard'`: Z-score normalization (mean=0, std=1)
- `'minmax'`: Min-Max scaling (range [0, 1])

## Output

The preprocessor provides detailed console output during processing:

```
============================================================
STARTING ML DATA PREPROCESSING PIPELINE
============================================================
Loading data from data.csv...
Loaded 200 rows and 5 columns

Splitting features and labels...
Label column: 'label'
Feature matrix X: (200, 4)
Label vector y: (200,)

Handling missing values using 'mean' strategy...
Found 10 NaN values
Missing values handled

Normalizing features using 'standard' method...
Features normalized

============================================================
PREPROCESSING COMPLETE!
============================================================
Final X shape: (200, 4)
Final y shape: (200,)
```

## Project Structure

```
ml-data-preprocessor/
├── data_preprocessor.py    # Main preprocessor class
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── examples/             # Example usage scripts
```

## Acknowledgments

- Built with NumPy and pandas
- Designed for seamless integration with scikit-learn and other ML libraries
- Perfect for data science education and rapid prototyping

