"""
Housing Price Prediction - Data Preprocessing Module

This module contains all preprocessing functions for the Ames Housing dataset.
It can be imported by all model notebooks to ensure consistent data preparation.

Author: Abel Varghese
Date: December 2025
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def load_data(train_path='train.csv', test_path=None):
    # Load training dataset from CSV file (test file is optional)
    train_df = pd.read_csv(train_path)
    
    print(f"Train shape: {train_df.shape}")
    
    if test_path:
        test_df = pd.read_csv(test_path)
        print(f"Test shape: {test_df.shape}")
    else:
        test_df = None
        print("No test file provided - will only create train/val/test split from training data")
    
    return train_df, test_df


def preprocess_data(train_df, test_df, val_size=0.15, test_size=0.15, random_state=42):
    # Complete 10-step preprocessing pipeline: outlier removal, log transform, imputation, feature engineering, encoding, scaling, splitting into train/val/test
    
    print("="*60)
    print("Starting Data Preprocessing Pipeline")
    print("="*60)
    
    # Step 1: Create working copies
    print("\n[1/10] Creating working copies...")
    train = train_df.copy()
    
    # Separate target variable
    y = train['SalePrice'].copy()
    train = train.drop(['SalePrice'], axis=1)
    
    # Store IDs
    train_id = train['Id']
    
    # Step 2: Handle outliers
    print("[2/10] Removing outliers...")
    outliers_idx = train[(train['GrLivArea'] > 4000) & (y < 300000)].index
    print(f"  Removing {len(outliers_idx)} outlier(s)")
    
    train = train.drop(outliers_idx)
    y = y.drop(outliers_idx)
    
    # Step 3: Transform target variable
    print("[3/10] Log-transforming target variable...")
    y_log = np.log1p(y)
    print(f"  Original skewness: {skew(y):.4f}")
    print(f"  Log-transformed skewness: {skew(y_log):.4f}")
    
    # Step 4: Process training data only (no Kaggle test set)
    print("[4/10] Processing training data...")
    ntrain = train.shape[0]
    all_data = train.copy()
    print(f"  Training data shape: {all_data.shape}")
    
    # Step 5: Handle missing values
    print("[5/10] Handling missing values...")
    
    # Features where NA means "None" or absence
    none_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                     'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                     'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                     'MasVnrType']
    
    for col in none_features:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna('None')
    
    # Features where NA should be 0
    zero_features = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea',
                     'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                     'BsmtFullBath', 'BsmtHalfBath']
    
    for col in zero_features:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna(0)
    
    # LotFrontage: fill with median by neighborhood
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # Fill remaining categorical with mode
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    
    # Fill remaining numeric with median
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if all_data[col].isnull().sum() > 0:
            all_data[col] = all_data[col].fillna(all_data[col].median())
    
    missing_count = all_data.isnull().sum().sum()
    print(f"  Missing values after imputation: {missing_count}")
    
    # Step 6: Feature engineering
    print("[6/10] Engineering new features...")
    
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    
    all_data['TotalBath'] = (all_data['FullBath'] + 
                             0.5 * all_data['HalfBath'] + 
                             all_data['BsmtFullBath'] + 
                             0.5 * all_data['BsmtHalfBath'])
    
    all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + 
                                all_data['3SsnPorch'] + 
                                all_data['EnclosedPorch'] + 
                                all_data['ScreenPorch'] + 
                                all_data['WoodDeckSF'])
    
    all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
    all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
    
    all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
    all_data['Has2ndFloor'] = (all_data['2ndFlrSF'] > 0).astype(int)
    all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
    all_data['HasBsmt'] = (all_data['TotalBsmtSF'] > 0).astype(int)
    all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
    
    print(f"  Created 10 new features")
    
    # Step 7: Encode quality features (ordinal)
    print("[7/10] Encoding ordinal features...")
    
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    quality_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                        'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    
    for col in quality_features:
        if col in all_data.columns:
            all_data[col] = all_data[col].map(quality_map)
    
    bsmt_exposure_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    all_data['BsmtExposure'] = all_data['BsmtExposure'].map(bsmt_exposure_map)
    
    bsmt_fin_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map(bsmt_fin_map)
    all_data['BsmtFinType2'] = all_data['BsmtFinType2'].map(bsmt_fin_map)
    
    functional_map = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 
                      'Min2': 6, 'Min1': 7, 'Typ': 8}
    all_data['Functional'] = all_data['Functional'].map(functional_map)
    
    garage_finish_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    all_data['GarageFinish'] = all_data['GarageFinish'].map(garage_finish_map)
    
    fence_map = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
    all_data['Fence'] = all_data['Fence'].map(fence_map)
    
    # Step 8: Handle skewed features
    print("[8/10] Correcting skewed features...")
    
    numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
    skew_features = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    high_skew = skew_features[abs(skew_features) > 0.75]
    
    print(f"  Transforming {len(high_skew)} highly skewed features")
    
    for feat in high_skew.index:
        all_data[feat] = np.log1p(all_data[feat])
    
    # Step 9: Drop highly correlated features
    print("[9/10] Removing multicollinearity...")
    
    correlated_features_to_drop = ['GarageArea', '1stFlrSF', 'TotRmsAbvGrd']
    all_data = all_data.drop(correlated_features_to_drop, axis=1, errors='ignore')
    print(f"  Dropped {len(correlated_features_to_drop)} highly correlated features")
    
    # Drop ID column
    all_data = all_data.drop(['Id'], axis=1, errors='ignore')
    
    # One-hot encode categorical variables
    categorical_cols = all_data.select_dtypes(include=['object']).columns
    print(f"  One-hot encoding {len(categorical_cols)} categorical features...")
    
    all_data = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)
    
    print(f"  Final feature count: {all_data.shape[1]}")
    
    # Step 10: Scale and split data
    print("[10/10] Scaling and splitting data...")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(all_data)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, 
                                   columns=all_data.columns, 
                                   index=all_data.index)
    
    # Three-way split: Train / Validation / Test from train.csv
    # First split: separate test set
    X_temp, X_test_internal, y_temp, y_test_internal = train_test_split(
        X_train_scaled, y_log, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation from remaining training data
    # Adjust val_size to be proportion of remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)
    print(f"Training set: {X_train_final.shape} (~{(1-val_size-test_size)*100:.0f}%)")
    print(f"Validation set: {X_val.shape} (~{val_size*100:.0f}%)")
    print(f"Test set (internal): {X_test_internal.shape} (~{test_size*100:.0f}%)")
    print("="*60)
    
    # Return None for Kaggle test set since it doesn't exist
    return X_train_final, X_val, y_train_final, y_val, X_test_internal, y_test_internal, None, None, scaler


def get_preprocessed_data(train_path='train.csv', test_path=None, 
                         val_size=0.15, test_size=0.15, random_state=42):
    # Convenience function that loads and preprocesses data in one call (returns train/val/test, no Kaggle test)
    train_df, test_df = load_data(train_path, test_path)
    return preprocess_data(train_df, test_df, val_size, test_size, random_state)


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing preprocessing pipeline...")
    print()
    
    X_train, X_val, y_train, y_val, X_test_internal, y_test_internal, X_kaggle_test, test_id, scaler = get_preprocessed_data()
    
    print("\nPreprocessing pipeline test successful!")
    print(f"All data ready for modeling (no Kaggle test set).")
