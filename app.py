# Import basic libraries
import pandas as pd
import numpy as np

# For visualization 
import matplotlib.pyplot as plt
import seaborn as sns

# For preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ML Model
from sklearn.ensemble import RandomForestRegressor

# For evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load dataset
df = pd.read_csv("crop_yield.csv")
# Clean text columns
df["Season"] = df["Season"].astype(str).str.strip().str.lower()
df["State"] = df["State"].astype(str).str.strip().str.lower()
df["Crop"] = df["Crop"].astype(str).str.strip().str.lower()


# Show first 5 rows
df.head()
# Dataset info
df.info()

# Check missing values
df.isnull().sum()

# Basic statistics
df.describe()
# Create encoders
crop_encoder = LabelEncoder()
season_encoder = LabelEncoder()
state_encoder = LabelEncoder()

# Encode columns
df["Crop"] = crop_encoder.fit_transform(df["Crop"])
df["Season"] = season_encoder.fit_transform(df["Season"])
df["State"] = state_encoder.fit_transform(df["State"])
# input
X = df.drop("Yield", axis=1)
model_features = X.columns.tolist()


# output
y = df["Yield"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

#Visualization
plt.figure()
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Yield")
plt.show()
importances = model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure()
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Feature Importance")
plt.show()

# Crop Recommendation System


def recommend_best_crop(user_input, model, all_crops, model_features):

    predictions = []

    for crop in all_crops:

        temp_data = user_input.copy()
        temp_data["Crop"] = crop

        
        temp_df = pd.DataFrame([temp_data])

        temp_df = temp_df[model_features]

        predicted_yield = model.predict(temp_df)[0]

        predictions.append((crop, predicted_yield))

    best_crop = max(predictions, key=lambda x: x[1])

    return best_crop, predictions


# Test with the User Input

all_crops = df["Crop"].unique()

user_data = {
    "Crop_Year": 2025,
    "Season": season_encoder.transform(["kharif"])[0],
    "State": state_encoder.transform(["andhra pradesh"])[0],
    "Area": 3.5,
    "Production": 2000,
    "Annual_Rainfall": 850,
    "Fertilizer": 150,
    "Pesticide": 50
}


# Run Recommendation

best_crop, all_predictions = recommend_best_crop(
    user_data,
    model,
    all_crops,
    model_features
)

best_crop_name = crop_encoder.inverse_transform([best_crop[0]])[0]

print("\nRecommended Crop:", best_crop_name)
print("Expected Yield:", round(best_crop[1], 2))


# Show Top 5 Crops
print("\nTop 5 Crop Recommendations:")

top_5 = sorted(all_predictions, key=lambda x: x[1], reverse=True)[:5]

for i, (crop, value) in enumerate(top_5, start=1):
    name = crop_encoder.inverse_transform([crop])[0]
    print(f"{i}. {name} → {round(value,2)}")

