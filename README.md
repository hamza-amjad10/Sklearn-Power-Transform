# Concrete Strength Prediction with Linear Regression

This project demonstrates the use of **Linear Regression** to predict the **compressive strength of concrete**.
It explores data preprocessing techniques, **skewness analysis**, and **power transformations** (Box-Cox and Yeo-Johnson) to improve model performance.

---

## 📂 Dataset

The dataset used is `12 concrete.csv`, which contains features related to the ingredients and properties of concrete, along with the target variable **Strength**.

---

## ⚙️ Workflow

1. **Data Preprocessing**

   * Split dataset into features (X) and target (Y).
   * Train-test split (80/20).
   * Skewness check for each feature using:

     * Histogram
     * Q-Q Plot
     * Skewness value

2. **Transformation**

   * Applied `PowerTransformer`:

     * **Box-Cox** (for strictly positive values) → R² ≈ 0.80
     * **Yeo-Johnson** (handles both positive & negative values) → R² ≈ 0.81

3. **Model Training**

   * Trained a **Linear Regression** model on transformed features.

4. **Evaluation**

   * Predictions on test set.
   * Evaluated with **R² score**.

---

## 📊 Results

* Box-Cox Transformation → R² ≈ 0.80
* Yeo-Johnson Transformation → R² ≈ 0.81

---

## 🚀 How to Run

# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib & Seaborn
* SciPy

---

## 📌 Notes

* Box-Cox requires strictly positive values, hence a small constant (`+0.00001`) was added to features when using it.
* Yeo-Johnson works for both positive and negative values, making it more flexible.

---

## ✨ Author

Created by **Hamza Amjad** as part of machine learning experiments.
