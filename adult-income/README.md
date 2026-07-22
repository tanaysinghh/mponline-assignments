# Adult Census Income Classification — Logistic Regression

**AI-ML Assignment**

## Objective
The US Census Bureau wants to predict whether an individual earns **more than \$50,000 per year** based on demographic and employment attributes. This project builds a **Logistic Regression** classifier to predict the `income` column (`<=50K` vs `>50K`) of the Adult Census Income dataset.

## Dataset
- **Source:** [Adult Census Income — Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income)
- **Original source:** UCI Machine Learning Repository (1994 US Census)
- **Rows:** 32,561
- **Features:** 14 (demographic and employment attributes)
- **Target:** `income` (`<=50K` / `>50K`)

> The dataset is **not** committed to this repository. Download `adult.csv` from Kaggle and place it in the project root before running the notebook.

## Libraries Used
- Python 3.10+
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` & `seaborn` — visualization
- `scikit-learn` — preprocessing, Logistic Regression model, evaluation metrics

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Methodology
1. **Data Understanding** — load the dataset, inspect the first five rows, and identify numerical features (`age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`, `hours.per.week`), categorical features (8 columns including `workclass`, `education`, `marital.status`, `occupation`), and the target (`income`).
2. **Data Preprocessing** — replace `?` placeholders with NaN, impute missing values in `workclass`, `occupation`, and `native.country` with the mode, map the target to 0/1, one-hot encode all categorical variables with `drop_first=True`, and perform a **stratified** 80/20 train-test split. Numerical features are standardized with `StandardScaler`.
3. **Model Development** — fit `sklearn.linear_model.LogisticRegression(max_iter=1000)` on the training set and predict on the test set.
4. **Model Evaluation** — score with Accuracy, Precision, Recall, F1-Score, and visualize a Confusion Matrix.
5. **Conclusion** — interpret the coefficients, discuss the main drivers of high income, and note the key limitation of Logistic Regression.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.8521 |
| Precision (>50K) | 0.7376 |
| Recall (>50K) | 0.5989 |
| F1-Score (>50K) | 0.6610 |

**Confusion Matrix:**
```
              Predicted
              <=50K   >50K
Actual <=50K   4611    334
       >50K     629    939
```

**Key observations:**
- Overall accuracy is around **85%**, but the dataset is imbalanced (~76% earn ≤\$50K), so accuracy alone overstates performance on the minority class.
- **Recall on the >50K class is only ~60%** — the model misses roughly 40% of actual high earners. Precision is higher (~74%), so when it does predict >50K it is usually right.
- **Marital status (married-civ-spouse), capital gain, education level, hours per week, and age** emerged as the strongest positive drivers of high income.

## Conclusion
Logistic Regression provides a strong baseline for census income classification with ~85% accuracy. The main drivers of predicting >\$50K income were **marital status, capital gains, education level, weekly hours worked, and age**. The main **limitation** is that Logistic Regression assumes a linear relationship between the features and the log-odds of the target — it cannot capture non-linear interactions (for example, education level combined with occupation). Non-linear models such as Random Forest and Gradient Boosting typically outperform it on this dataset.

## Repository Structure
```
.
├── Assignment.ipynb          # Main notebook with all 5 tasks
├── README.md                 # This file
└── .gitignore
```

## How to Run
```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <repo-name>

# 2. Download adult.csv from Kaggle and place it in this folder

# 3. Launch Jupyter
jupyter notebook Assignment.ipynb
```

---
**Author:** Tanay Singh — 23BCE11211
