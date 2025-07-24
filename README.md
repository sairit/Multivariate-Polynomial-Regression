# Polynomial Regression Implementation from Scratch

**Author:** Sai Yadavalli  
**Version:** 2.0

A comprehensive implementation of polynomial regression with k-fold cross-validation, built from scratch using NumPy for mathematical computations and featuring automated model selection capabilities.

## Overview

This project implements polynomial regression without relying on scikit-learn or other machine learning frameworks, demonstrating mastery of linear algebra, optimization theory, and statistical validation techniques. The implementation includes automatic data splitting, cross-validation, model comparison, and predictive capabilities with comprehensive visualization.

## Mathematical Foundation

### Polynomial Feature Transformation
The core concept extends linear regression to capture non-linear relationships by transforming input features:

```
X_poly = [1, x, x^2, x^3, ..., x^d]
```

Where `d` is the polynomial degree, creating a feature matrix that enables linear methods to model non-linear patterns.

### Normal Equation Solution
The implementation uses the closed-form solution for optimal weights:

```
W = (X^T * X)^(-1) * X^T * Y
```

This analytical approach provides exact solutions without iterative optimization, leveraging:
- Matrix pseudo-inverse for numerical stability
- Direct linear algebra computation
- Guaranteed convergence to global minimum

### Cost Function
The model minimizes mean squared error:

```
J(W) = (1/m) * ||XW - Y||^2
```

Where:
- `m` is the number of training examples
- `||·||^2` represents the squared L2 norm
- The cost measures average prediction error

## Features

- **Pure NumPy Implementation**: Direct mathematical computation without ML libraries
- **Automated K-Fold Cross-Validation**: 5-fold validation with automatic data splitting
- **Multi-Degree Analysis**: Simultaneous training and comparison of polynomial degrees 1 through n
- **Bias-Variance Analysis**: Training vs validation cost tracking for overfitting detection
- **Matrix Operations**: Efficient linear algebra using NumPy's optimized routines
- **Visualization Suite**: Comprehensive plotting for model comparison and data exploration
- **Predictive Interface**: Interactive testing with trained models
- **File Management**: Automated directory creation and CSV handling

## Key Components

### Core Methods

#### `polynomial_features(X, degree)` - Feature Engineering
Transforms input data into polynomial feature space by computing powers up to the specified degree, enabling linear methods to capture polynomial relationships.

#### `costs(X, Y, w=None)` - Model Evaluation
Computes both the optimal weights using the normal equation and the resulting cost function value. Supports both training (weight computation) and testing (given weights) modes.

#### `matrix(trainFeature, testFeature, train)` - Data Preparation
Extracts and formats feature and target variables from pandas DataFrames into NumPy matrices suitable for mathematical operations.

### Cross-Validation System

#### `split_data()` - K-Fold Data Partitioning
Implements stratified 5-fold cross-validation by:
- Randomly sampling 20% for each fold
- Creating complementary 80% training sets
- Maintaining statistical properties across folds
- Saving splits as CSV files for reproducibility

#### `training(featureA, featureB, degree)` - Model Training Pipeline
Orchestrates the complete training process:
- Iterates through polynomial degrees 1 to n
- Trains on each fold's training set
- Validates on corresponding validation set
- Aggregates results for statistical analysis

### Analysis and Visualization

#### `train_table()` - Performance Analysis
Provides comprehensive model evaluation including:
- Cost comparison across polynomial degrees
- Training vs validation performance plots
- Average cost computation for each degree
- Statistical significance assessment

#### `plot_regression(featureA, featureB)` - Model Visualization
Creates publication-quality plots showing:
- Original data points
- Fitted polynomial curves for each degree
- Comparative visualization of model complexity

#### `calculate_regression(x, y, degree)` - Prediction Generation
Computes polynomial regression predictions and displays learned parameters, enabling model interpretation and coefficient analysis.

## Technical Implementation

### Numerical Stability
- **Pseudo-inverse**: Uses `np.linalg.pinv()` for robust matrix inversion
- **Matrix Conditioning**: Handles near-singular matrices gracefully
- **Vectorized Operations**: Leverages NumPy's optimized linear algebra routines

### Memory Management
- **Efficient Storage**: Dynamic array growth for cross-validation results
- **File Organization**: Structured directory hierarchy for data management
- **DataFrame Integration**: Seamless pandas-NumPy interoperability

### Error Handling
- **Matrix Dimensionality**: Automatic shape validation and correction
- **File I/O**: Robust CSV reading with error reporting
- **User Input**: Validation and sanitization of interactive inputs

## Usage

### Training New Models
```python
# Initialize and train model
model = PolyRegression("Year", "Temperature", degree=5, to_train=True)

# Analyze cross-validation results
model.train_table()

# Visualize fitted curves
model.plot_regression("Year", "Temperature")
```

### Using Pre-trained Models
```python
# Load model with known weights
model = PolyRegression("Year", "Temperature", degree=3, to_train=False)
model.set_weights([52.71, -0.085, 0.0014, -5.18e-06])

# Make predictions
model.test()
```

### Interactive Analysis
```python
# Complete analysis pipeline
model.load_data()           # Load dataset
model.split_data()          # Create cross-validation folds
model.training(x, y, deg)   # Train multiple polynomial degrees
model.train_table()         # Display results and plots
```

## Cross-Validation Results

The implementation provides detailed statistical analysis:

### Model Comparison
- **Training Costs**: Measure of fit quality on training data
- **Validation Costs**: Indicator of generalization performance
- **Degree Analysis**: Optimal complexity selection through bias-variance tradeoff

### Overfitting Detection
- **Divergence Monitoring**: Training vs validation cost comparison
- **Complexity Penalties**: Higher degree polynomial performance tracking
- **Statistical Significance**: Cross-fold variance analysis

## Performance Monitoring

### Cost Function Tracking
- Degree-wise cost computation
- Fold-by-fold performance variation
- Statistical summary statistics

### Visual Analysis
- Training/validation cost curves
- Polynomial regression overlays
- Data distribution visualization

## Requirements

```
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.3.0
```

## Educational Value

This implementation demonstrates:

### Mathematical Concepts
- **Linear Algebra**: Matrix operations, pseudo-inverse, eigenvalue stability
- **Polynomial Mathematics**: Feature transformation, basis functions
- **Optimization Theory**: Closed-form solutions vs iterative methods
- **Statistical Validation**: Cross-validation, bias-variance decomposition

### Software Engineering
- **Modular Design**: Clean separation of concerns
- **File Management**: Automated directory structure and data persistence
- **User Interface**: Interactive command-line testing environment
- **Documentation**: Comprehensive docstrings and code organization

### Data Science Pipeline
- **Data Preprocessing**: Automated splitting and matrix conversion
- **Model Selection**: Systematic degree comparison
- **Performance Evaluation**: Multiple metrics and visualization
- **Deployment**: Prediction interface with trained models

## Future Enhancements

- [ ] Regularization techniques (Ridge, Lasso)
- [ ] Feature scaling and normalization
- [ ] Multivariate polynomial regression
- [ ] Bootstrap validation methods
- [ ] Automated hyperparameter tuning
- [ ] Model persistence and serialization

---

This implementation serves as a comprehensive demonstration of polynomial regression theory and practice, showcasing both mathematical rigor and practical software development skills in the machine learning domain.
