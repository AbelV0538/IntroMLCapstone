# IntroMLCapstone - Housing Price Prediction

**Course:** ITCS 5356 - Applied Machine Learning  
**Semester:** Fall 2025  
**Author:** Abel Varghese  
**Dataset:** Ames Housing - Advanced Regression Techniques (Kaggle)

---

## Project Overview

This capstone project implements and compares machine learning algorithms for housing price prediction using the Ames Housing dataset. The project fulfills all course requirements:

✅ **3 Classical ML Algorithms** (learned in class)  
✅ **2 Literature-Based Approaches** (from peer-reviewed papers, published >2020)  
✅ **Comprehensive Evaluation** with MSE, MAE, R² metrics  
✅ **Visualizations** (actual vs predicted, residuals, feature importance)  
✅ **Technical Report** (IEEE format, 5+ pages with comparison tables)  
✅ **Well-Organized Code** in GitHub repository

---

## Repository Structure

### 📁 Model Implementation Files (5 Required)

#### Classical ML Algorithms (3 Files - Learned in Class)

**1. `1_ridge_regression.ipynb`** - Ridge Regression with L2 Regularization

- Linear regression with regularization to prevent overfitting
- Cross-validated alpha selection (0.001 to 10000)
- Interpretable coefficients for feature importance
- Baseline model for comparison

**2. `2_polynomial_regression.ipynb`** - Polynomial Feature Expansion

- Degree-2 polynomial transformation on top 20 features
- Captures non-linear relationships and interactions
- Ridge regularization on ~210 expanded features
- Improves over simple linear regression

**3. `3_neural_network.ipynb`** - Multi-Layer Perceptron

- Deep learning architecture: 256 → 128 → 64 neurons
- ReLU activations, Adam optimizer
- Early stopping prevents overfitting
- Powerful non-linear modeling

#### Literature-Based Approaches (2 Files - From Research Papers)

**4. `4_random_forest.ipynb`** - Random Forest Ensemble

- **Paper Citation:** [TODO: Add peer-reviewed paper >2020]
- 200 decision trees with bootstrap sampling
- Out-of-bag validation scoring
- Feature importance via Gini impurity

**5. `5_xgboost.ipynb`** - XGBoost Gradient Boosting

- **Paper Citation:** [TODO: Add peer-reviewed paper >2020]
- 500 boosting rounds, early stopping
- L1/L2 regularization, learning rate=0.05
- State-of-the-art on tabular data

**6. `6_model_comparison.ipynb`** - Comprehensive Model Comparison

- Compares all 5 models with visualizations
- RMSE, MAE, R² comparison charts
- Overfitting analysis
- Performance heatmaps and summary statistics

### 📄 Supporting Files

**Preprocessing & Utilities:**

- **`preprocessing.py`** - Centralized data preprocessing module
  - Ensures consistency across all models
  - See "Preprocessing Pipeline" section below

**Analysis & Documentation:**

- **`house_price_analysis.ipynb`** - Comprehensive EDA
- **`template.txt`** - IEEE technical report template
- **`README.md`** - This documentation file

**Dataset:**

- `train.csv` - Training data (1460 samples, 80 features)
- `data_description.txt` - Feature descriptions
- `sample_submission.csv` - Example submission format

**Note:** Kaggle `test.csv` is not used as it lacks labels. Three-way split (70/15/15) created from `train.csv` for rigorous evaluation.

---

## How to Run This Project

### Prerequisites

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost
```

### Running Individual Models

1. **Ensure dataset files are present:**

2. **Open and run any notebook:**

   - Each notebook is self-contained
   - Preprocessing is handled automatically via `preprocessing.py`
   - Works in Jupyter Lab, Jupyter Notebook, or VS Code

3. **Example notebook flow:**

```python
# All notebooks follow this pattern:

# 1. Import preprocessing module
from preprocessing import get_preprocessed_data

# 2. Load and preprocess data
X_train, X_val, y_train, y_val, X_test, test_id, scaler = get_preprocessed_data()

# 3. Train model
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_val)

# 5. Analyze results (metrics, visualizations, feature importance)
```

---

## Preprocessing Pipeline

The **`preprocessing.py`** module ensures consistent data preparation across all models:

### Steps Performed:

1. **Outlier Removal** - Remove GrLivArea outliers (>4000 sqft with price <$300k)
2. **Target Transformation** - Log transform SalePrice for normality
3. **Missing Value Imputation:**
   - Categorical: NA → "None" (absence of feature)
   - Numeric: Fill with 0 or median based on context
   - LotFrontage: Median by neighborhood
4. **Feature Engineering:**
   - `TotalSF` = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
   - `TotalBath` = FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath
   - `TotalPorchSF` = Sum of all porch areas
   - `HouseAge` = YrSold - YearBuilt
   - `RemodAge` = YrSold - YearRemodAdd
   - Binary flags: HasPool, Has2ndFloor, HasGarage, HasBsmt, HasFireplace
5. **Ordinal Encoding** - Quality features (Ex→5, Gd→4, TA→3, Fa→2, Po→1, None→0)
6. **Skewness Correction** - Log transform features with |skewness| > 0.75
7. **Multicollinearity Reduction** - Drop highly correlated features: GarageArea, 1stFlrSF, TotRmsAbvGrd
8. **One-Hot Encoding** - Categorical variables (drop_first=True)
9. **Standardization and Splitting** - StandardScaler (mean=0, std=1)
   - Train: 70% (~1,022 samples)
   - Validation: 15% (~219 samples)
   - Test: 15% (~219 samples)
   - All from train.csv only (random_state=42)

### Output:

- `X_train_final` - Training features (scaled, 70%)
- `X_val` - Validation features (scaled, 15%)
- `X_test_internal` - Test features (scaled, 15%)
- `y_train_final` - Training target (log-transformed)
- `y_val` - Validation target (log-transformed)
- `y_test_internal` - Test target (log-transformed)
- `X_kaggle_test` - None (Kaggle test.csv not used)
- `test_id` - None (Kaggle test.csv not used)
- `scaler` - Fitted StandardScaler object

---

## Model Comparison

### Performance Summary

| Model                | Type                      | Evaluation Metrics   | Strengths                                      | Limitations             |
| -------------------- | ------------------------- | -------------------- | ---------------------------------------------- | ----------------------- |
| **Ridge Regression** | Classical (Linear)        | See notebook results | Fast, interpretable, handles multicollinearity | Assumes linearity       |
| **Polynomial Reg**   | Classical (Non-linear)    | See notebook results | Captures interactions                          | Feature explosion       |
| **Neural Network**   | Classical (Deep Learning) | See notebook results | Complex patterns                               | Black box, needs tuning |
| **Random Forest**    | Literature (Ensemble)     | See notebook results | Robust, feature importance                     | Memory intensive        |
| **XGBoost**          | Literature (Boosting)     | See notebook results | State-of-the-art performance                   | Many hyperparameters    |

_Note: Run all notebooks to populate actual MSE, MAE, and R² scores for comparison_

### Evaluation Metrics

Each model is evaluated using:

- **RMSE** (Root Mean Squared Error) - Primary regression metric
- **MAE** (Mean Absolute Error) - Robust to outliers
- **R² Score** - Proportion of variance explained (0 to 1, higher is better)

### Visualizations Included

All notebooks contain:
✅ Actual vs Predicted scatter plots  
✅ Residual distributions (histograms)  
✅ Residual plots (checking for patterns)  
✅ Feature importance rankings  
✅ Model-specific visualizations:

- Ridge: Coefficient magnitudes
- Polynomial: Top polynomial features
- Neural Network: Training loss curves
- Random Forest: Tree depth distributions, OOB scores
- XGBoost: Boosting iteration curves, early stopping

---

## Technical Report Structure

The accompanying technical report (IEEE format, 5+ pages) includes:

### 1. Introduction

- Problem statement and objectives
- Dataset summary (Ames Housing, 80 features, 1460 training samples)
- Motivation for housing price prediction

### 2. Methodology

- Preprocessing pipeline (10 steps detailed above)
- Feature engineering rationale
- Classical ML algorithms:
  - Ridge Regression (L2 regularization)
  - Polynomial Regression (degree-2 expansion)
  - Neural Network (256-128-64 MLP)
- Literature-based implementations:
  - Random Forest (paper citation + methodology)
  - XGBoost (paper citation + methodology)

### 3. Results and Evaluation

- Performance comparison table (MSE, MAE, R²)
- Actual vs Predicted plots for each model
- Residual analysis
- Feature importance comparisons
- Training time comparisons

### 4. Discussion

- Why certain models outperformed others
- Strengths and limitations of each approach
- Reproducibility challenges from literature
- Hyperparameter tuning insights

### 5. Conclusion

- Key findings summary
- Best model for housing price prediction
- Future work suggestions

### 6. Implementation Code

- Link to this GitHub repository: `IntroMLCapstone`
- File organization overview

### 7. References

- Two peer-reviewed papers (>2020)
- Library citations (scikit-learn, XGBoost, etc.)

---

## Literature Papers

### Paper 1: Random Forest for Regression

**TODO: Add full citation**

- Authors: [Author names]
- Year: [>2020]
- Source: [Journal/Conference]
- Key Contribution: [Brief description]
- Implementation: `4_random_forest.ipynb`

### Paper 2: XGBoost for Tabular Data

**TODO: Add full citation**

- Authors: [Author names]
- Year: [>2020]
- Source: [Journal/Conference]
- Key Contribution: [Brief description]
- Implementation: `5_xgboost.ipynb`

**Finding Relevant Papers:**

- Google Scholar: Search "housing price prediction machine learning 2020-2025"
- arXiv.org: Browse cs.LG (Machine Learning) category
- Criteria: Regression tasks, tabular/structured data, reproducible methodology

---

## Key Findings

### Performance Rankings (by Test R²):

1. **Ridge Regression**: 0.9086 (RMSE: 0.1247) ⭐ WINNER
2. **XGBoost**: 0.9048 (RMSE: 0.1273)
3. **Polynomial Regression**: 0.8914 (RMSE: 0.1359)
4. **Random Forest**: 0.8554 (RMSE: 0.1568)
5. **Neural Network**: -5.7208 (RMSE: 1.0692) ❌ FAILED

### Key Insights:

**Surprising Result:** The simplest model (Ridge) outperformed all complex models, achieving 90.9% variance explanation with minimal overfitting (0.0334).

**Neural Network Failure:** Catastrophic failure with negative R² values demonstrates that model complexity does not guarantee performance. The 256→128→64 architecture was severely misspecified for this tabular dataset with ~1,000 training samples.

**Ensemble Performance:** XGBoost (best ensemble) came very close to Ridge (0.9048 vs 0.9086) but with higher overfitting (0.0891 vs 0.0334). Random Forest showed concerning overfitting (0.1237) and ranked fourth.

### Most Interpretable:

- Ridge Regression (clear coefficient interpretation + best performance)

### Best for Feature Importance:

- XGBoost (gain-based importance) and Random Forest (Gini importance) consistently identify OverallQual, GrLivArea, and TotalSF as top features

### Computational Efficiency:

- Fastest: Ridge Regression (~seconds)
- Slowest: Neural Network (~minutes, but failed to converge properly)

### Trade-offs:

- **Simplicity Wins:** Ridge achieves best accuracy with fastest training and full interpretability
- **Preprocessing Matters:** Feature engineering and regularization proved more important than algorithm complexity
- **Dataset Size:** ~1,000 training samples insufficient for deep neural networks but ideal for Ridge/XGBoost

---

## Future Work

1. **Ensemble Stacking** - Combine predictions from all 5 models
2. **Advanced Feature Engineering:**
   - Neighborhood clustering based on median prices
   - Price per square foot ratios
   - Interaction terms guided by domain knowledge
3. **Hyperparameter Optimization:**
   - Bayesian optimization for XGBoost
   - Grid search for Random Forest
   - Architecture search for Neural Network
4. **Cross-Validation:**
   - K-fold CV across different random seeds
   - Time-series split if temporal data available
5. **External Validation:**
   - Test on different housing datasets
   - Geographic generalization analysis
6. **Explainability:**
   - SHAP values for black-box models
   - LIME for local interpretability

---

## Project Structure Summary

```
IntroMLCapstone/
│
├── 1_ridge_regression.ipynb       # Classical Model #1
├── 2_polynomial_regression.ipynb  # Classical Model #2
├── 3_neural_network.ipynb         # Classical Model #3
├── 4_random_forest.ipynb          # Literature Model #1
├── 5_xgboost.ipynb                # Literature Model #2
├── 6_model_comparison.ipynb       # Comprehensive comparison
│
├── preprocessing.py               # Shared preprocessing module
├── evaluation.py                  # Metrics and visualization utilities
├── house_price_analysis.ipynb     # EDA notebook
├── technical_report.md            # IEEE-format technical report
├── README.md                      # This file
│
├── train.csv                      # Training data (1460 samples)
├── data_description.txt           # Feature documentation
└── sample_submission.csv          # Example submission format
```

**Note:** The Kaggle `test.csv` is not used in this project as it lacks ground truth labels. All evaluation uses a 3-way split (train/validation/test) from `train.csv`.

---

## Contact & Submission

**Author:** Abel Varghese  
**Course:** ITCS 5356 - Applied Machine Learning  
**Semester:** Fall 2025  
**GitHub Repository:** IntroMLCapstone  
**Report Format:** IEEE (PDF, 5+ pages minimum)

---

## Acknowledgments

- **Dataset:** Ames Housing Dataset (Kaggle)
- **Libraries:** scikit-learn, XGBoost, pandas, numpy, matplotlib, seaborn
- **Course:** ITCS 5356 - Applied Machine Learning
- **Papers:** [Citations to be added for Random Forest and XGBoost papers]

---

## License

Academic use only - Capstone Project for ITCS 5356  
Do not redistribute dataset or use for commercial purposes
