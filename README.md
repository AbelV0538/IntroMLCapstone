# IntroMLCapstone - Housing Price Prediction

**Course:** ITCS 5356 - Applied Machine Learning  
**Semester:** Fall 2025  
**Author:** Abel Varghese  
**Dataset:** Ames Housing - Advanced Regression Techniques (Kaggle)

---

## File Descriptions

### Jupyter Notebooks

**`1_ridge_regression.ipynb`**  
Ridge Regression implementation with L2 regularization and cross-validated alpha selection.

**`2_polynomial_regression.ipynb`**  
Polynomial regression with degree-2 feature expansion on top 20 features.

**`3_neural_network.ipynb`**  
Multi-layer perceptron (256→128→64 architecture) with ReLU activations and early stopping.

**`4_random_forest.ipynb`**  
Random Forest ensemble with 200 trees using bootstrap aggregation.

**`5_xgboost.ipynb`**  
XGBoost gradient boosting with 500 rounds, L1/L2 regularization, and subsampling.

**`6_model_comparison.ipynb`**  
Comprehensive comparison of all 5 models with visualizations, metrics tables, and performance analysis.

**`house_price_analysis.ipynb`**  
Exploratory data analysis of the Ames Housing dataset.

### Python Modules

**`preprocessing.py`**  
Centralized preprocessing pipeline with 10 steps including outlier removal, feature engineering, encoding, scaling, and 3-way data split (70% train / 15% validation / 15% test).

**`evaluation.py`**  
Utility functions for model evaluation metrics and comparison visualizations.

### Documentation

**`technical_report.md`**  
IEEE-format technical report (1500+ words) with methodology, results, discussion, and conclusions.

**`README.md`**  
This file - project overview and file descriptions.

### Data Files

**`train.csv`**  
Training dataset (1,460 samples, 80 features) from Kaggle Ames Housing competition.

**`data_description.txt`**  
Detailed descriptions of all 80 features in the dataset.

**`sample_submission.csv`**  
Example submission format for Kaggle competition.

---

**Note:** All models use `random_state=42` for reproducibility. The Kaggle `test.csv` is not used as it lacks ground truth labels; instead, a 3-way split from `train.csv` provides train/validation/test sets.
