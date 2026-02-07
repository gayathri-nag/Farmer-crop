# 🌾 Crop Yield Prediction and Recommendation System using Machine Learning

## 📖 Introduction
Agriculture plays a crucial role in India's economy. However, farmers often face uncertainty in choosing crops due to changing climate conditions, soil quality, and resource availability.

This project uses Machine Learning to analyze historical agricultural data and predict crop yield. It also provides crop recommendations to help farmers maximize productivity and reduce risk.

---

## 🎯 Problem Statement
Due to unpredictable environmental conditions, farmers find it difficult to estimate crop production in advance. Traditional methods rely on experience rather than data-driven decisions.

This project aims to:
- Predict crop yield accurately
- Identify important influencing factors
- Recommend the best crops for given conditions
- Support smart farming decisions

---

## 📊 Dataset Description
A real-world agricultural dataset was used for this project. It contains 19,000+ records with the following attributes:

| Feature | Description |
|---------|-------------|
| Crop | Type of crop |
| Crop Year | Year of cultivation |
| Season | Season of farming |
| State | State location |
| Area | Land area (hectares) |
| Production | Total production |
| Annual Rainfall | Rainfall (mm) |
| Fertilizer | Fertilizer usage |
| Pesticide | Pesticide usage |
| Yield | Crop yield (Target) |

Missing values were checked and no major missing data was found.

---

## ⚙️ Technologies and Tools
- Programming Language: Python
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-learn
- Platform: VS Code, GitHub

---

## 🧠 Machine Learning Model

### Selected Algorithm: Random Forest Regressor

Random Forest was selected because:
- It handles non-linear relationships well
- Reduces overfitting
- Works efficiently with large datasets
- Provides feature importance

It is suitable for real-world agricultural data.

---

## 🔄 Project Workflow

1. Data Collection
2. Data Exploration and Cleaning
3. Categorical Encoding
4. Feature Selection
5. Train-Test Split
6. Model Training
7. Performance Evaluation
8. Visualization
9. Recommendation System

---

## 📈 Data Preprocessing

- Checked for missing values
- Removed unnecessary columns
- Converted categorical features using Label Encoding
- Normalized numerical data where required
- Split dataset into 80% training and 20% testing

---

## 📊 Model Evaluation

The model performance was evaluated using the following metrics:

### 🔹 Mean Absolute Error (MAE)

MAE = (1/n) Σ |y - ŷ|

Measures average prediction error.

### 🔹 Root Mean Squared Error (RMSE)

RMSE = √[(1/n) Σ (y - ŷ)²]

Penalizes large errors.

### 🔹 R² Score (Coefficient of Determination)

R² = 1 - (SSres / SStot)

Measures goodness of fit.

---

## 📌 Results

The trained model achieved:

- MAE: Low error value
- RMSE: Acceptable variance
- R² Score: ~0.90

This indicates strong prediction capability.

---

## 📉 Data Visualization

The following visualizations were created:

### 1️⃣ Actual vs Predicted Yield
Shows accuracy of predictions.

### 2️⃣ Yield Distribution
Displays yield spread in dataset.

### 3️⃣ Feature Importance
Identifies major influencing factors.

(Refer screenshots below)

---

## 🌱 Crop Recommendation System

A recommendation system was implemented using the trained model.

### Working:
- User enters soil and weather parameters
- System predicts yield for all crops
- Top 5 crops are ranked
- Best crop is recommended

This helps farmers select suitable crops.


