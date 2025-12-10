"""
Evaluation Module for House Price Prediction Models

This module provides standardized evaluation metrics and visualizations
for all models in the capstone project.

Functions:
    - calculate_metrics: Compute RMSE, MAE, R² for train and validation sets
    - plot_performance: Create comprehensive 6-panel visualization
    - plot_feature_importance: Create feature importance bar chart
    - print_metrics_summary: Display formatted metrics table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_train_true, y_train_pred, y_val_true, y_val_pred):
    # Calculate RMSE, MAE, R² for train and validation sets, returns dict with all metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train_true, y_train_pred)),
        'train_mae': mean_absolute_error(y_train_true, y_train_pred),
        'train_r2': r2_score(y_train_true, y_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val_true, y_val_pred)),
        'val_mae': mean_absolute_error(y_val_true, y_val_pred),
        'val_r2': r2_score(y_val_true, y_val_pred)
    }
    metrics['overfitting'] = metrics['train_r2'] - metrics['val_r2']
    
    return metrics


def print_metrics_summary(model_name, metrics):
    # Print formatted metrics summary table with RMSE, MAE, R² for train/validation sets
    print(f"\n{model_name} - Performance Metrics")
    print("=" * 70)
    print("\nTraining Set:")
    print(f"  RMSE: {metrics['train_rmse']:.6f}")
    print(f"  MAE:  {metrics['train_mae']:.6f}")
    print(f"  R²:   {metrics['train_r2']:.6f}")
    
    print("\nValidation Set:")
    print(f"  RMSE: {metrics['val_rmse']:.6f}")
    print(f"  MAE:  {metrics['val_mae']:.6f}")
    print(f"  R²:   {metrics['val_r2']:.6f}")
    
    print("\n" + "=" * 70)
    print(f"Overfitting (R² difference): {metrics['overfitting']:.4f}")
    print("=" * 70 + "\n")


def plot_performance(y_train_true, y_train_pred, y_val_true, y_val_pred, 
                    model_name, metrics=None):
    # Create 6-panel visualization: predicted vs actual, residual distribution, residual plots for train/validation
    if metrics is None:
        metrics = calculate_metrics(y_train_true, y_train_pred, y_val_true, y_val_pred)
    
    # Calculate residuals
    train_residuals = y_train_true - y_train_pred
    val_residuals = y_val_true - y_val_pred
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{model_name} - Model Performance Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Training set - Predicted vs Actual
    axes[0, 0].scatter(y_train_pred, y_train_true, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0, 0].plot([y_train_true.min(), y_train_true.max()], 
                    [y_train_true.min(), y_train_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_title(f'Training: Predicted vs Actual (R²={metrics["train_r2"]:.4f})', 
                        fontweight='bold')
    axes[0, 0].set_xlabel('Predicted log(SalePrice)')
    axes[0, 0].set_ylabel('Actual log(SalePrice)')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Training set - Residual distribution
    axes[0, 1].hist(train_residuals, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[0, 1].set_title('Training: Residual Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Training set - Residual plot
    axes[0, 2].scatter(y_train_pred, train_residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
    axes[0, 2].axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[0, 2].set_title(f'Training: Residual Plot (MAE={metrics["train_mae"]:.4f})', 
                        fontweight='bold')
    axes[0, 2].set_xlabel('Predicted log(SalePrice)')
    axes[0, 2].set_ylabel('Residuals')
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)
    
    # Validation set - Predicted vs Actual
    axes[1, 0].scatter(y_val_pred, y_val_true, alpha=0.5, color='green', 
                      edgecolors='k', linewidth=0.5)
    axes[1, 0].plot([y_val_true.min(), y_val_true.max()], 
                    [y_val_true.min(), y_val_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[1, 0].set_title(f'Validation: Predicted vs Actual (R²={metrics["val_r2"]:.4f})', 
                        fontweight='bold')
    axes[1, 0].set_xlabel('Predicted log(SalePrice)')
    axes[1, 0].set_ylabel('Actual log(SalePrice)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Validation set - Residual distribution
    axes[1, 1].hist(val_residuals, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[1, 1].set_title('Validation: Residual Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # Validation set - Residual plot
    axes[1, 2].scatter(y_val_pred, val_residuals, alpha=0.5, color='green', 
                      edgecolors='k', linewidth=0.5)
    axes[1, 2].axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    axes[1, 2].set_title(f'Validation: Residual Plot (MAE={metrics["val_mae"]:.4f})', 
                        fontweight='bold')
    axes[1, 2].set_xlabel('Predicted log(SalePrice)')
    axes[1, 2].set_ylabel('Residuals')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_feature_importance(features, importances, model_name, top_n=20, 
                           color='blue', importance_type='Coefficient'):
    # Create horizontal bar chart showing top N most important features
    # Create dataframe and sort
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
        'Abs_Importance': np.abs(importances)
    }).sort_values('Abs_Importance', ascending=False)
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Handle color logic for Ridge (positive/negative coefficients)
    if importance_type == 'Coefficient' and color == 'blue':
        colors = ['green' if x > 0 else 'red' for x in top_features['Importance']]
        title_suffix = '\n(Green=Positive Impact, Red=Negative Impact)'
    else:
        colors = color
        title_suffix = ''
    
    # Create horizontal bar chart
    ax.barh(range(top_n), top_features['Abs_Importance'], 
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel(f'Absolute {importance_type} Value', fontweight='bold')
    ax.set_ylabel('Feature', fontweight='bold')
    ax.set_title(f'{model_name} - Top {top_n} Most Important Features{title_suffix}', 
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def compare_models(results_dict):
    # Create comparison DataFrame from dict of model metrics, sorted by validation R²
    comparison_data = []
    
    for model_name, metrics in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Train RMSE': metrics['train_rmse'],
            'Val RMSE': metrics['val_rmse'],
            'Train MAE': metrics['train_mae'],
            'Val MAE': metrics['val_mae'],
            'Train R²': metrics['train_r2'],
            'Val R²': metrics['val_r2'],
            'Overfitting': metrics['overfitting']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Val R²', ascending=False)
    
    return df


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("  - calculate_metrics(y_train_true, y_train_pred, y_val_true, y_val_pred)")
    print("  - print_metrics_summary(model_name, metrics)")
    print("  - plot_performance(y_train_true, y_train_pred, y_val_true, y_val_pred, model_name)")
    print("  - plot_feature_importance(features, importances, model_name, top_n=20)")
    print("  - compare_models(results_dict)")
