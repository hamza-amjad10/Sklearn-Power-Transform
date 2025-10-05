import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("12 concrete.csv")

# Split features (X) and target (Y)
X = df.drop("Strength", axis=1)
Y = df["Strength"]

# Train-test split (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Check skewness of each column
for col in df:
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df[col], kde=True, ax=ax[0])
    ax[0].set_title(f'Histogram of {col}')  # column name as title
    stats.probplot(df[col], dist="norm", plot=ax[1])
    ax[1].set_title(f"Q-Q plot of {col}")
    plt.show()
    print(f"columns name is: {col} and skew value is :{df[col].skew()}")

# --- Data Transformation ---
# Box-Cox works only for positive values (gave r2 ≈ 0.80 in testing)
# Yeo-Johnson works for both positive & negative values (gave r2 ≈ 0.81 here)


# pt=PowerTransformer(method="box-cox") r2_score is  0.80

# X_train_tranformed=pt.fit_transform(X_train+0.00001)
# X_test_tranformed=pt.transform(X_test+0.00001)


pt = PowerTransformer(method="yeo-johnson")

# Fit & transform train data, and transform test data
X_train_transformed = pt.fit_transform(X_train)
X_test_transformed = pt.transform(X_test)

# --- Model Training ---
lr = LinearRegression()
lr.fit(X_train_transformed, Y_train)

# --- Prediction ---
y_predict = lr.predict(X_test_transformed)

# --- Evaluation ---
print(y_predict)  # predicted values (can be commented out if not needed)
print(f"R² score: {r2_score(Y_test, y_predict)}")
