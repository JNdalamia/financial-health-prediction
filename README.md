# Financial Health Prediction — Zindi

## Project Overview

This project develops a machine learning classification model to predict the **financial health of individuals/business owners** using financial, demographic, business, insurance, savings, and financial-access information.

The project was completed as part of the **Zindi Financial Inclusion in Africa** competition.

**Competition:** Financial Inclusion in Africa — Financial Health Prediction  
**Platform:** Zindi  
**Problem Type:** Multiclass Classification  
**Target Variable:** `Target`  
**Target Classes:** `Low`, `Medium`, `High`

The goal is to build a reliable machine learning model that can classify financial health and potentially support data-driven financial inclusion initiatives.

---

## Competition

This project uses data from the Zindi competition:

**Financial Inclusion in Africa**

Competition page:

https://zindi.world/competitions/financial-inclusion-in-africa

---

## Objectives

The main objectives of this project were to:

1. Understand the financial health dataset through exploratory data analysis.
2. Investigate relationships between demographic, financial, business, and insurance variables and financial health.
3. Identify and appropriately handle missing values.
4. Prepare categorical and numerical variables for machine learning.
5. Establish a Logistic Regression baseline.
6. Compare multiple machine learning algorithms.
7. Address class imbalance using class weighting where appropriate.
8. Tune the strongest model using cross-validation.
9. Evaluate models using **Macro F1**, Accuracy, and Weighted F1.
10. Interpret model predictions and identify influential features.
11. Train the selected final model using the complete labelled dataset.
12. Generate predictions for the official Zindi test dataset.
13. Create and validate the final Zindi submission file.

---

## Dataset

The training dataset contains:

- **9,618 observations**
- **39 columns**
- A mixture of numerical and categorical variables
- Four countries represented in the dataset
- A three-class target variable:
  - `Low`
  - `Medium`
  - `High`

### Example Variables

The dataset includes variables covering areas such as:

- Country
- Owner age
- Personal income
- Business expenses
- Business turnover
- Business age
- Mobile money usage
- Insurance
- Loan accounts
- Savings behaviour
- Access to financial services
- Internet access
- Debit cards
- Informal lending
- Medical and funeral insurance
- Friends and family savings

The project also uses the provided variable-definition file to understand the meaning of the dataset variables.

---

# Project Workflow

## 1. Data Loading

The project begins by loading the training dataset using pandas.

```python
fh_df = pd.read_csv("data/financial_health.csv")
```

Basic dataset checks were performed to understand:

- Dataset dimensions
- Column names
- Data types
- First and last records
- Summary statistics
- Missing values

---

## 2. Exploratory Data Analysis

Exploratory Data Analysis was performed before modelling to understand the structure and characteristics of the dataset.

### Target Distribution

The target variable was examined to determine the distribution of the three financial-health classes.

```python
fh_df["Target"].value_counts()
```

The target classes are:

- Low
- Medium
- High

The analysis showed that the classes are not perfectly balanced, making **Macro F1** an important evaluation metric.

---

## 3. Country Analysis

The distribution of observations across countries was examined.

Country-level financial-health distributions were also calculated using cross-tabulation.

```python
pd.crosstab(
    fh_df["country"],
    fh_df["Target"],
    normalize="index"
) * 100
```

This helped identify differences in financial-health patterns between countries.

---

## 4. Numerical Variable Analysis

Important numerical variables were examined using descriptive statistics and visualizations.

Examples include:

- `owner_age`
- `personal_income`
- `business_expenses`
- `business_turnover`
- `business_age_years`
- `business_age_months`

Histograms were used to understand distributions, while boxplots were used to compare numerical variables across financial-health classes.

For example:

- Personal income vs financial health
- Business turnover vs financial health
- Business expenses vs financial health

A logarithmic scale was used for income visualization because financial variables showed substantial skewness.

---

## 5. Categorical Variable Analysis

Categorical variables were analysed to investigate their relationship with financial health.

Examples included:

- Mobile money
- Insurance
- Loan accounts
- Funeral insurance
- Other financial-access variables

For each categorical variable, the percentage of observations in each financial-health class was calculated.

This helped identify categories associated with higher or lower proportions of the `High` financial-health class.

---

# 6. Missing Value Analysis

Missing values were investigated across the dataset.

A missing-value summary was created containing:

- Number of missing observations
- Percentage of missing observations

Missingness was also compared against the target variable to determine whether missing values appeared to be associated with financial-health classes.

For example:

```python
pd.crosstab(
    fh_df["has_loan_account"].isna(),
    fh_df["Target"],
    normalize="index"
)
```

This analysis helped inform the preprocessing strategy.

---

# 7. Data Cleaning

Categorical values were checked for inconsistent labels.

One issue identified was the use of curly apostrophes in categorical values.

These were normalized:

```python
fh_df_clean[col] = fh_df_clean[col].str.replace(
    "’",
    "'",
    regex=False
)
```

This prevented visually identical categories from being treated as different values.

The `ID` column was removed from the modelling dataset because it is an identifier rather than a predictive feature.

---

# 8. Train/Test Split

The labelled dataset was split into training and testing sets using an 80/20 split.

```python
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean,
    y_clean,
    test_size=0.2,
    random_state=42,
    stratify=y_clean
)
```

### Split Strategy

- Training set: 80%
- Test set: 20%
- `random_state`: 42
- Stratification: enabled

Stratification ensured that the target-class proportions were approximately maintained in both datasets.

---

# 9. Preprocessing Pipeline

A Scikit-learn `ColumnTransformer` was used to create a reproducible preprocessing pipeline.

## Numerical Features

Numerical variables were processed using:

1. Median imputation
2. Standard scaling

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

Median imputation was selected because it is less sensitive to extreme values than mean imputation.

---

## Categorical Features

Categorical variables were processed using:

1. Missing-value imputation
2. One-hot encoding

```python
Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="missing"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])
```

`handle_unknown="ignore"` ensures that unseen categories in the test data do not cause the prediction pipeline to fail.

---

# 10. Machine Learning Models

Several classification algorithms were evaluated.

The models included:

### Logistic Regression

Used as the baseline model.

### Balanced Logistic Regression

A second Logistic Regression model was trained using:

```python
class_weight="balanced"
```

This was used to investigate the impact of class imbalance.

### Random Forest

A Random Forest classifier was trained using multiple decision trees.

### Decision Tree

A standalone Decision Tree classifier was evaluated.

### K-Nearest Neighbours

KNN was included as a distance-based classification approach.

### Support Vector Machine

An SVM classifier was evaluated using class weighting.

### Gradient Boosting

Gradient Boosting was evaluated as an ensemble boosting approach.

### Extra Trees

An Extra Trees classifier was also evaluated as an ensemble method.

---

# 11. Model Evaluation

Models were evaluated using three main metrics:

### Accuracy

Measures the overall percentage of correctly classified observations.

### Macro F1

Calculates F1 independently for each class and then averages the scores.

Macro F1 was selected as the primary model-selection metric because it gives equal importance to:

- Low
- Medium
- High

This is particularly useful when the target classes are imbalanced.

### Weighted F1

Calculates F1 while weighting each class according to its number of observations.

---

# 12. Model Comparison

The final experiment compared the following models:

| Model | Evaluation |
|---|---|
| Decision Tree | Accuracy: 0.8254, Macro F1: 0.7491, Weighted F1: 0.8255 |
| KNN | Accuracy: 0.7713, Macro F1: 0.6288, Weighted F1: 0.7521 |
| SVM | Accuracy: 0.8321, Macro F1: 0.7610, Weighted F1: 0.8356 |
| Gradient Boosting | Accuracy: 0.8758, Macro F1: 0.8064, Weighted F1: 0.8687 |
| Extra Trees | Accuracy: 0.8701, Macro F1: 0.7973, Weighted F1: 0.8652 |
| Tuned Random Forest | Accuracy: 0.856, Macro F1: 0.7929, Weighted F1: 0.8556 |

The models were trained using the same preprocessing pipeline to make the comparison consistent.

The model with the highest **Macro F1 (Gradient Boosting)** was selected as the best-performing model.

```python
best_model_name = result_df.iloc[0]["Model"]

best_pipeline = trained_models[best_model_name]
```

---

# 13. Random Forest Hyperparameter Tuning

Random Forest was further optimized using `RandomizedSearchCV`.

The parameters explored included:

- Number of estimators
- Maximum tree depth
- Minimum samples required to split a node
- Minimum samples per leaf
- Maximum number of features

Example search space:

```python
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"]
}
```

The search used:

- 15 randomized combinations
- 5-fold cross-validation
- Macro F1 as the optimization metric
- Random state = 42

```python
RandomizedSearchCV(
    estimator=rf_pipeline_clean,
    param_distributions=param_grid,
    n_iter=15,
    scoring="f1_macro",
    cv=5,
    random_state=42,
    n_jobs=-1
)
```

---

# 14. Model Interpretation

Model interpretation was performed to understand which variables contributed most strongly to predictions.

Random Forest feature importance was initially examined using:

```python
rf_model.feature_importances_
```

Permutation importance was also calculated using **Macro F1** as the scoring metric.

```python
permutation_importance(
    rf_pipeline_clean,
    X_test_clean,
    y_test_clean,
    scoring="f1_macro",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)
```

Permutation importance provides a more model-agnostic indication of how much predictive performance changes when individual features are shuffled.

---

# 15. Final Model

The highest-performing model based on the model comparison was selected as the final model.

The selected pipeline was cloned before final training:

```python
final_model = clone(best_pipeline)
```

The final model was then trained using **all available labelled training data**:

```python
final_model.fit(X_clean, y_clean)
```

This allowed the final model to learn from the maximum amount of labelled information before generating competition predictions.

---

# 16. Zindi Test Predictions

The official Zindi test dataset was loaded and subjected to the same categorical cleaning process used for the training data.

```python
test_data = pd.read_csv("data/Test.csv")
```

The `ID` column was excluded from model features.

Predictions were then generated:

```python
zindi_predictions = final_model.predict(X_zindi_test)
```

The prediction distribution was also checked to ensure that the model produced valid target classes.

---

# 17. Submission File

The final submission was created using the required Zindi format:

```text
ID,Target
```

The predictions were combined with the original test IDs:

```python
submission = pd.DataFrame({
    "ID": test_data["ID"],
    "Target": zindi_predictions
})
```

Several validation checks were performed before saving the file.

These checks confirmed that:

- Column names were correct
- Column order was correct
- Number of rows matched the sample submission
- IDs matched the official test dataset
- Only valid target labels were generated
- No predictions were missing

The final submission was saved as:

```text
outputs/submission.csv
```

---

# Results

The project evaluates models primarily using **Macro F1**, with Accuracy and Weighted F1 used as supporting metrics.

The final model-selection results are generated in the notebook using:

```python
result_df
```

To reproduce the exact results, run the complete notebook from the data-loading stage through the model-comparison stage.

> **Note:** The final Zindi leaderboard score should be added here after submitting `submission.csv` to the competition.

### Final Competition Score

**Zindi Public Leaderboard Score:** 0.88407343

---

# Key Machine Learning Decisions

Several decisions were made during the project:

### Macro F1 as the primary metric

Because the target contains three classes and is not perfectly balanced, Macro F1 was prioritized over accuracy alone.

### Stratified splitting

Stratification was used to preserve the target-class distribution between training and testing datasets.

### Pipeline-based preprocessing

All preprocessing was incorporated into Scikit-learn pipelines to reduce the risk of inconsistent transformations and data leakage.

### Median imputation

Numerical missing values were handled using median imputation.

### Explicit categorical missing category

Missing categorical observations were represented as `"missing"` rather than simply removing observations.

### One-hot encoding

Categorical variables were converted into machine-readable numerical features using OneHotEncoder.

### Class weighting

Class-balanced models were tested to determine whether improving minority-class performance could improve Macro F1.

### Cross-validation

Randomized hyperparameter search with 5-fold cross-validation was used to improve Random Forest performance.

---

# Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Development Environment

- Jupyter Notebook
- Visual Studio Code

### Competition Platform

- Zindi

---

# Python Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
tabulate
```

---

# Project Structure

A recommended GitHub repository structure is:

```text
Financial_Health_Prediction/
│
├── data/
│   ├── financial_health.csv
│   ├── Test.csv
│   ├── SampleSubmission.csv
│   └── VariableDefinitions.csv
│
├── notebooks/
│   └── financial_health_prediction.ipynb
│
├── outputs/
│   └── submission.csv
│
│
├── figures/
│   ├── target_distribution.png
│   ├── country_distribution.png
│   ├── financial_health_by_country.png
│   ├── logistic_regression_confusion_matrix.png
    |── random_forest_confusion_matrix.png
│   └── top_20_random_forest_features.png
│
├── .gitignore
└── README.md
```

**Important:** Competition datasets may have redistribution restrictions. Check the Zindi competition rules before committing the raw datasets to a public GitHub repository. If required, keep the data locally and document where it can be obtained.

---

# Reproducibility

To reproduce this project:

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Financial_Health_Prediction.git
```

### 2. Navigate to the project

```bash
cd Financial_Health_Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tabulate jupyter
```

### 6. Run the notebook

```bash
jupyter notebook
```

Open the project notebook and execute the cells sequentially.

---

# Lessons Learned

This project provided practical experience in:

- Exploratory data analysis
- Multiclass classification
- Handling imbalanced target variables
- Missing-value analysis
- Feature preprocessing
- One-hot encoding
- Scikit-learn pipelines
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Random Forest modelling
- Feature importance
- Permutation importance
- Model evaluation
- Competition submission preparation

A key lesson was that **accuracy alone is not sufficient for evaluating a multiclass classification problem with imbalanced classes**. Macro F1 provides a more balanced view of performance across the different financial-health categories.

---

# Future Improvements

Potential improvements to the project include:

1. More extensive hyperparameter optimization.
2. Experimenting with XGBoost, LightGBM, or CatBoost.
3. Feature engineering for financial ratios and business characteristics.
4. More detailed treatment of outliers.
5. Testing different approaches to missing-value handling.
6. Calibration of class probabilities.
7. SHAP-based model interpretation.
8. Ensemble modelling or stacking.
9. More extensive cross-validation.
10. Threshold and probability analysis for the three financial-health classes.
11. Further analysis of country-specific patterns.
12. Additional feature-selection experiments.

---

# Conclusion

This project demonstrates a complete end-to-end machine learning workflow for a real-world financial-health classification problem.

The workflow covers:

**Data → EDA → Data Cleaning → Missing Values → Preprocessing → Baseline Model → Model Comparison → Hyperparameter Tuning → Model Interpretation → Final Training → Prediction → Zindi Submission**

The project emphasizes reproducibility, appropriate evaluation metrics, pipeline-based preprocessing, and practical model interpretation.

It also demonstrates how machine learning can be applied to financial inclusion data to identify patterns associated with different levels of financial health.

---

## Author

**Jason Kevin Ndalamia**

ICT & Data Professional | Machine Learning & Data Analytics

Kenya

- GitHub: https://github.com/JNdalamia
- LinkedIn: https://linkedin.com/in/jason-ndalamia
- DEV Community: https://dev.to/jason_ndalamia

---

## Acknowledgements

- **Zindi** for providing the competition platform and dataset.
- The organizers of the **Financial Health Prediction** challenge.
- The open-source Python and Scikit-learn communities for the tools used throughout the project.

---

## License

This repository is intended primarily for educational and portfolio purposes.

Please refer to the original Zindi competition terms and dataset license before redistributing competition data or derived datasets.