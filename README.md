# Customer Churn Prediction using Logistic Regression

## Project Overview

This project is a **Machine Learning classification project** that predicts whether a telecom customer is likely to churn.

The project uses the **Logistic Regression** algorithm to classify customers into two categories:

- `0` → Customer does not churn
- `1` → Customer churns

The notebook performs data loading, data inspection, missing-value handling, visualization, categorical encoding, feature/target separation, train-test splitting, model training, prediction, and accuracy evaluation.

---

## Project Objective

The main objective of this project is to use customer information such as:

- Gender
- Senior citizen status
- Partner/dependent information
- Tenure
- Phone services
- Internet services
- Online security
- Online backup
- Device protection
- Technical support
- Streaming services
- Contract type
- Paperless billing
- Payment method
- Monthly charges
- Total charges

to predict whether a customer will churn.

---

## Project Structure

```text
Customer-Churn-Logistic-Regression/
│
├── customer_churn_logistic_regression.ipynb
├── customer_churn_.csv
├── README.md
└── requirements.txt
```

### Files

#### `customer_churn_logistic_regression.ipynb`

The main Jupyter Notebook containing the complete machine learning workflow.

#### `customer_churn_.csv`

The customer churn dataset used by the notebook.

#### `README.md`

Documentation explaining the project, dataset, code, dependencies, and execution instructions.

#### `requirements.txt`

Contains the Python libraries required to run the project.

---

# Dataset

The dataset is loaded using:

```python
df = pd.read_csv('/content/customer_churn_.csv')
```

The original dataset contains:

```text
7043 rows
21 columns
```

The dataset contains customer information and the target variable `Churn`.

---

## Dataset Columns

The original dataset contains the following columns:

```text
customerID
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
Churn
```

---

## Target Variable

The target variable is:

```text
Churn
```

It represents whether a customer has left the company.

The target is converted into numerical values using `LabelEncoder`.

Conceptually:

```text
No  → 0
Yes → 1
```

---

# Technologies Used

The project uses Python and the following libraries:

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Visualization |
| Scikit-learn | Machine learning |
| Jupyter Notebook | Development environment |
| Google Colab | Optional execution environment |

---

# Dependencies

The project requires the following Python packages:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

# Installation

You can install the required packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Then install all dependencies:

```bash
pip install -r requirements.txt
```

---

# Project Workflow

The complete workflow is:

```text
Dataset
   ↓
Load Data
   ↓
Inspect Data
   ↓
Check Missing Values
   ↓
Check Duplicate Values
   ↓
Explore Numerical Variables
   ↓
Convert TotalCharges to Numeric
   ↓
Remove Missing Values
   ↓
Remove customerID
   ↓
Encode Categorical Variables
   ↓
Separate Features and Target
   ↓
Train/Test Split
   ↓
Logistic Regression
   ↓
Model Training
   ↓
Prediction
   ↓
Accuracy Evaluation
```

---

# Code Details

## 1. Import Libraries

The project starts by importing the required libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Purpose

- `pandas` → DataFrame operations and CSV loading
- `numpy` → Numerical calculations
- `matplotlib` → Plotting
- `seaborn` → Visualization

---

# 2. Load Dataset

The dataset is loaded using Pandas:

```python
df = pd.read_csv('/content/customer_churn_.csv')
```

The notebook then displays the dataset:

```python
df
```

---

# 3. Inspect Dataset

The project checks the dataset information:

```python
df.info()
```

This helps identify:

- Number of rows
- Number of columns
- Data types
- Non-null values

The project initially contains:

```text
7043 rows × 21 columns
```

---

# 4. Check Missing Values

Missing values are checked using:

```python
df.isnull().sum().sum()
```

The notebook also checks the individual columns:

```python
df.isnull().sum()
```

---

# 5. Check Duplicate Records

Duplicate records are checked using:

```python
df.duplicated().sum()
```

This determines whether duplicate customer records exist.

---

# 6. Check Dataset Columns

The column names are displayed using:

```python
df.columns
```

This allows the project to identify the available customer attributes.

---

# 7. Data Visualization

The project creates box plots for numerical columns:

```python
for col in df.columns:
    if df[col].dtype != 'object':
        plt.boxplot(df[col])
        plt.title(col)
        plt.show()
```

This is used to visually inspect numerical variables and possible outliers.

---

# 8. Convert `TotalCharges` to Numeric

The `TotalCharges` column is converted into a numeric data type:

```python
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)
```

The `errors='coerce'` option converts values that cannot be interpreted as numbers into missing values.

The data is checked again:

```python
df.info()
```

and:

```python
df.isnull().sum()
```

---

# 9. Remove Missing Values

Rows containing missing values are removed:

```python
df.dropna(inplace=True)
```

After preprocessing, the dataset contains approximately:

```text
7032 rows
```

---

# 10. Remove `customerID`

The customer ID is an identifier and is not used as a machine learning feature.

Therefore, it is removed:

```python
df.drop(['customerID'], axis=1, inplace=True)
```

---

# 11. Encode Categorical Variables

Machine learning algorithms require numerical input.

The project uses `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
```

This converts categorical/object columns into numerical values.

For example, categorical values such as:

```text
Yes
No
```

are converted into numerical representations.

---

# 12. Separate Features and Target

The target column is `Churn`.

The features are created using:

```python
x = df.drop('Churn', axis=1)
```

The target is:

```python
y = df['Churn']
```

Therefore:

```text
X = Input features
Y = Churn target
```

---

# 13. Train/Test Split

The dataset is divided into training and testing data.

The project uses:

```python
from sklearn.model_selection import train_test_split
```

The split is:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.80,
    random_state=0
)
```

This means approximately:

```text
80% → Training data
20% → Testing data
```

The notebook has approximately:

```text
Training samples: 5625
Testing samples: 1407
```

---

# 14. Logistic Regression

The machine learning algorithm used is Logistic Regression.

It is imported using:

```python
from sklearn.linear_model import LogisticRegression
```

The model is created:

```python
model = LogisticRegression()
```

---

# 15. Train the Model

The model is trained using:

```python
model.fit(x_train, y_train)
```

The model learns the relationship between customer characteristics and churn.

---

# 16. Make Predictions

After training, predictions are generated on the test dataset:

```python
y_pred = model.predict(x_test)
```

The predictions are stored in:

```text
y_pred
```

---

# 17. Evaluate Model Accuracy

The project imports the evaluation metrics:

```python
from sklearn.metrics import *
```

The model accuracy is calculated using:

```python
accuracy_score(y_test, y_pred)
```

Accuracy represents the percentage of test observations that were classified correctly.

---

# Complete Code

The main code from the notebook can be summarized as:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/content/customer_churn_.csv')

# Inspect dataset
df
df.info()

# Check missing values
df.isnull().sum().sum()

# Check duplicates
df.duplicated().sum()

# Display columns
df.columns

# Visualize numerical columns
for col in df.columns:
    if df[col].dtype != 'object':
        plt.boxplot(df[col])
        plt.title(col)
        plt.show()

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'],
    errors='coerce'
)

# Check information
df.info()

# Check missing values
df.isnull().sum()

# Remove missing rows
df.dropna(inplace=True)

# Check again
df.isnull().sum()

# Remove customer ID
df.drop(['customerID'], axis=1, inplace=True)

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Separate features and target
x = df.drop('Churn', axis=1)
y = df['Churn']

# Train/test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.80,
    random_state=0
)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# Train model
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Display predictions
y_pred

# Model evaluation
from sklearn.metrics import *

accuracy_score(y_test, y_pred)
```

---

# Important Model Note

The notebook reports a **Logistic Regression convergence warning** during model training.

The model uses:

```python
model = LogisticRegression()
```

For a future improved version, the number of iterations could be increased:

```python
model = LogisticRegression(max_iter=1000)
```

Feature scaling could also be added before training.

---

# Possible Improvements

The current project is a basic machine-learning implementation. It can be improved by adding:

### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
```

### Better preprocessing

Use:

```python
ColumnTransformer
Pipeline
OneHotEncoder
StandardScaler
```

instead of fitting `LabelEncoder` separately to each categorical feature.

### Additional evaluation metrics

Add:

```python
accuracy_score()
precision_score()
recall_score()
f1_score()
confusion_matrix()
classification_report()
roc_auc_score()
```

### Hyperparameter tuning

Use:

```python
GridSearchCV
```

to find better Logistic Regression parameters.

### Model comparison

Compare Logistic Regression with models such as:

```text
Decision Tree
Random Forest
XGBoost
SVM
K-Nearest Neighbors
```

---

# Limitations

The current project has some limitations:

1. The dataset path is hard-coded:

```python
/content/customer_churn_.csv
```

2. Categorical encoders are not saved for future predictions.

3. Numerical features are not scaled.

4. The notebook reports a Logistic Regression convergence warning.

5. The project currently evaluates the model primarily using accuracy.

6. There is no production-ready prediction pipeline.

---

# How to Run the Project

## Google Colab

1. Open the `.ipynb` file in Google Colab.
2. Upload `customer_churn_.csv`.
3. Make sure the dataset is available at:

```text
/content/customer_churn_.csv
```

4. Run the notebook cells from top to bottom.

---

## Local Jupyter Notebook

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
customer_churn_logistic_regression.ipynb
```

If your CSV is in the same directory as the notebook, change:

```python
df = pd.read_csv('/content/customer_churn_.csv')
```

to:

```python
df = pd.read_csv('customer_churn_.csv')
```

---

# Requirements

Create a file named:

```text
requirements.txt
```

with:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install using:

```bash
pip install -r requirements.txt
```

---

# Project Summary

| Property | Details |
|---|---|
| Project | Customer Churn Prediction |
| Machine Learning Type | Supervised Learning |
| Problem Type | Binary Classification |
| Algorithm | Logistic Regression |
| Dataset | Telecom Customer Churn |
| Initial Records | 7,043 |
| Initial Columns | 21 |
| Cleaned Records | 7,032 |
| Target | `Churn` |
| Train/Test Split | 80/20 |
| Preprocessing | Missing-value removal + Label Encoding |
| Evaluation | Accuracy |
| Language | Python |
| Environment | Jupyter / Google Colab |

---

# Conclusion

This project demonstrates a complete basic machine-learning workflow for **customer churn prediction**.

The process starts with raw customer data, performs data cleaning and preprocessing, converts categorical information into numerical values, separates the features and target, divides the data into training and testing sets, trains a Logistic Regression model, generates predictions, and evaluates the model using accuracy.

The project can be further improved with feature scaling, better categorical encoding, additional evaluation metrics, hyperparameter tuning, and a complete production-ready preprocessing pipeline.
