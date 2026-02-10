import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error


def convert_numbers(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()

        try:
            if x.endswith("K"):
                return float(x[:-1]) * 1_000
            elif x.endswith("M"):
                return float(x[:-1]) * 1_000_000
            else:
                return float(x)
        except:
            return np.nan

    return x


df = pd.read_csv("backloggd_games.csv")

features = ["Playing", "Wishlist", "Plays"]
target = "Reviews"

for col in features + [target]:
    df[col] = df[col].apply(convert_numbers)


df["engagement"] = df["Playing"] / (df["Plays"] + 1)

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=features + ["engagement", target])

X = df[features + ["engagement"]]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=80,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=80,
        learning_rate=0.1,
        random_state=42
    )
}

results = {}
best_model = None
best_score = -np.inf


for name, model in models.items():

    if name == "Linear Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results[name] = (r2, rmse)

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name


print("\n=== RESULTADOS ===")

for name, (r2, rmse) in results.items():
    print(f"{name:18} -> R2: {r2:.4f} | RMSE: {rmse:.2f}")

print("\nEntrenando modelo final con dataset completo...")

if best_name == "Linear Regression":
    best_model.fit(scaler.fit_transform(X), y)
else:
    best_model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump((best_model, scaler, best_name), f)

print(f"\n✅ Mejor modelo guardado: {best_name}")
