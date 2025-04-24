# Credit Risk Prediction for LendingClub Loans

## Dashboard: https://public.tableau.com/app/profile/maulik/viz/CreditRiskSnapshot/CreditRiskSnapshot

## Overview

This project develops a machine learning model to predict loan defaults using the LendingClub dataset. The model helps financial institutions assess credit risk by identifying potential defaults, supporting risk mitigation and regulatory compliance (e.g., IFRS 9). Using a Kaggle dataset with 759,338 loans (7% default rate), I achieved a 0.768 ROC-AUC, 0.70 recall for defaults, and calculated key risk metrics: average LGD of 0.513 and EAD of $14,708.

## Dataset

The [dataset](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans/data) contains loans issued by LendingClub from 2016 to 2017, sourced from Kaggle, with 759,338 records and a 7% default rate (52,060 defaults). It includes 67 features such as loan amount, interest rate, borrower income, debt-to-income ratio, and categorical variables (e.g., loan grade, purpose). The target variable `default` is binary (1 for default, 0 for non-default).

## Project Details

The goal is to predict whether a loan will default, a binary classification task. Key steps include:

- **Exploratory Data Analysis (EDA)**: Identified feature importance (e.g., `grade_C` and `grade_int_rate` were top predictors) and addressed multicollinearity by dropping redundant features.
- **Feature Engineering**: Created log-transformed features (e.g., `loan_amnt_log`, `annual_inc_log`) to handle skewness, and derived `grade_int_rate` (grade ordinal × interest rate) to capture risk interactions.
- **Handling Imbalance**: Used class weights and SMOTE to address the 7% default rate, though SMOTE introduced noise, leading to reliance on threshold tuning.
- **Model Training**: Used a LightGBM model with hyperparameter tuning via `RandomizedSearchCV` (60 fits over 3 folds). Best parameters: `n_estimators=500`, `max_depth=5`, `learning_rate=0.05`, `num_leaves=63`, `subsample=1.0`, `colsample_bytree=0.6`, `min_child_samples=100`, `class_weight=None`.
- **Evaluation**: Tested on 151,868 samples (10,412 defaults, 141,456 non-defaults), achieving 0.768 ROC-AUC and 0.70 recall for defaults at a threshold of 0.076 (optimized for max F1 with recall >0.7). Precision for defaults was 0.14 (86% false positives).
- **Risk Metrics**: Calculated LGD (1 - recovery rate) at 0.513 and EAD (using `funded_amnt`) at $14,708, providing actionable insights for credit risk management.
- **Challenges**: Attempted IFRS 9 staging and stress testing but faced issues with feature mismatches (`pd_features` vs. `pd_features_refined`) and dropped columns, suggesting future improvements with a cleaner subset.

## Key Achievements

- **Model Performance**: Achieved 0.768 ROC-AUC and 0.70 recall for default detection, balancing risk identification with operational feasibility.
- **Risk Metrics**: Quantified average LGD at 0.513 (51.3% loss on defaults) and EAD at $14,708, supporting financial risk assessment.
- **Practical Insights**: Demonstrated a robust approach for credit risk modeling, suitable for regulatory frameworks like IFRS 9, with potential for further optimization using cleaner data subsets.
