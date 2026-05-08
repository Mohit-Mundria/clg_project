"""
KisanAI - ML Model Training Script
Trains and saves:
  1. Crop Recommendation Model (Random Forest)
  2. Fertilizer Recommendation Model (Gradient Boosting)

Run this script ONCE before starting the server:
    python ml_training/train_models.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "backend" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print(" KisanAI — ML Model Training Script")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════════
# 1. CROP RECOMMENDATION MODEL
# ═══════════════════════════════════════════════════════════════════
def generate_crop_dataset(n_samples: int = 2200) -> pd.DataFrame:
    """
    Generate a realistic synthetic crop recommendation dataset.
    Based on the well-known Kaggle Crop Recommendation Dataset structure.
    Features: N, P, K, temperature, humidity, ph, rainfall
    Label: crop name (22 crops)
    """
    np.random.seed(42)

    # Each crop has ideal ranges for its features
    crop_profiles = {
        "rice":        dict(N=(80,100),  P=(40,60),   K=(40,55),  T=(22,28), H=(80,90), pH=(5.5,6.5), R=(200,300)),
        "wheat":       dict(N=(80,100),  P=(40,60),   K=(40,55),  T=(15,25), H=(55,75), pH=(6.0,7.5), R=(60,110)),
        "maize":       dict(N=(70,90),   P=(50,70),   K=(50,65),  T=(20,28), H=(60,75), pH=(5.5,7.0), R=(60,120)),
        "chickpea":    dict(N=(30,50),   P=(60,80),   K=(75,100), T=(18,26), H=(14,22), pH=(5.5,7.0), R=(60,100)),
        "kidneybeans": dict(N=(18,30),   P=(55,75),   K=(18,30),  T=(18,26), H=(18,25), pH=(5.5,7.0), R=(80,130)),
        "pigeonpeas":  dict(N=(16,25),   P=(55,75),   K=(16,22),  T=(25,32), H=(45,65), pH=(5.5,7.0), R=(60,100)),
        "mothbeans":   dict(N=(18,25),   P=(43,60),   K=(22,32),  T=(27,34), H=(47,60), pH=(3.5,6.5), R=(50,80)),
        "mungbean":    dict(N=(18,26),   P=(43,58),   K=(40,50),  T=(25,32), H=(82,90), pH=(6.2,7.2), R=(60,100)),
        "blackgram":   dict(N=(38,50),   P=(65,80),   K=(18,28),  T=(27,33), H=(64,75), pH=(6.0,7.5), R=(65,100)),
        "lentil":      dict(N=(16,24),   P=(60,78),   K=(24,35),  T=(16,24), H=(64,75), pH=(6.0,7.0), R=(35,60)),
        "pomegranate": dict(N=(15,22),   P=(5,15),    K=(35,45),  T=(20,30), H=(84,95), pH=(5.5,7.2), R=(80,150)),
        "banana":      dict(N=(96,115),  P=(74,90),   K=(45,58),  T=(25,30), H=(74,88), pH=(6.0,7.5), R=(100,200)),
        "mango":       dict(N=(15,22),   P=(5,12),    K=(30,45),  T=(24,32), H=(46,60), pH=(5.5,7.5), R=(50,100)),
        "grapes":      dict(N=(18,25),   P=(18,28),   K=(38,50),  T=(20,27), H=(80,90), pH=(5.5,7.0), R=(50,100)),
        "watermelon":  dict(N=(98,118),  P=(14,22),   K=(18,28),  T=(25,33), H=(84,92), pH=(6.0,7.0), R=(40,80)),
        "muskmelon":   dict(N=(98,118),  P=(14,22),   K=(18,28),  T=(27,34), H=(88,95), pH=(6.0,7.0), R=(20,45)),
        "apple":       dict(N=(18,28),   P=(18,25),   K=(8,15),   T=(15,22), H=(88,95), pH=(5.5,7.0), R=(80,130)),
        "orange":      dict(N=(18,28),   P=(15,22),   K=(8,15),   T=(22,28), H=(88,94), pH=(5.5,7.0), R=(60,110)),
        "papaya":      dict(N=(48,58),   P=(58,72),   K=(48,62),  T=(25,33), H=(90,95), pH=(5.5,7.0), R=(100,200)),
        "coconut":     dict(N=(18,25),   P=(5,12),    K=(28,38),  T=(26,33), H=(90,96), pH=(5.5,8.0), R=(100,200)),
        "cotton":      dict(N=(115,138), P=(42,58),   K=(40,55),  T=(24,30), H=(77,88), pH=(5.8,7.5), R=(60,120)),
        "jute":        dict(N=(66,85),   P=(42,58),   K=(38,55),  T=(25,35), H=(70,85), pH=(6.0,7.0), R=(170,250)),
        "coffee":      dict(N=(98,118),  P=(26,38),   K=(26,38),  T=(22,28), H=(54,68), pH=(6.0,7.0), R=(150,250)),
    }

    records = []
    per_crop = n_samples // len(crop_profiles)

    for crop, ranges in crop_profiles.items():
        for _ in range(per_crop):
            record = {
                "N":           np.random.uniform(*ranges["N"]),
                "P":           np.random.uniform(*ranges["P"]),
                "K":           np.random.uniform(*ranges["K"]),
                "temperature": np.random.uniform(*ranges["T"]),
                "humidity":    np.random.uniform(*ranges["H"]),
                "ph":          np.random.uniform(*ranges["pH"]),
                "rainfall":    np.random.uniform(*ranges["R"]),
                "label":       crop,
            }
            records.append(record)

    df = pd.DataFrame(records)
    # Add slight noise
    for col in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        df[col] += np.random.normal(0, df[col].std() * 0.05, len(df))
        df[col] = df[col].clip(0)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train_crop_model():
    print("\n[Step 1] Training Crop Recommendation Model (Random Forest)")
    print("-" * 50)

    df = generate_crop_dataset(2200)
    print(f"   Dataset size: {len(df)} samples, {df['label'].nunique()} crops")

    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].values
    y = df["label"].values

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   [OK] Test Accuracy: {accuracy * 100:.2f}%")

    # Save artifacts
    joblib.dump(model, MODEL_DIR / "crop_model.pkl")
    joblib.dump(encoder, MODEL_DIR / "crop_label_encoder.pkl")
    joblib.dump(scaler, MODEL_DIR / "crop_scaler.pkl")
    print(f"   [Saved] -> backend/models/crop_model.pkl")
    print(f"   [Saved] -> backend/models/crop_label_encoder.pkl")
    print(f"   [Saved] -> backend/models/crop_scaler.pkl")

    return accuracy


# ═══════════════════════════════════════════════════════════════════
# 2. FERTILIZER RECOMMENDATION MODEL
# ═══════════════════════════════════════════════════════════════════
SOIL_TYPES = ["sandy", "loamy", "black", "red", "clayey", "alluvial", "laterite"]
CROP_TYPES = ["rice", "wheat", "maize", "sugarcane", "cotton", "pulses", "vegetables", "fruits", "oilseeds", "barley"]
FERTILIZERS = ["Urea", "DAP", "MOP", "10-26-26", "17-17-17", "20-20", "28-28", "14-35-14"]


def assign_fertilizer(nitrogen, phosphorus, potassium, crop_type, soil_type):
    """Rule-based fertilizer assignment for training data."""
    n_low = nitrogen < 50
    p_low = phosphorus < 25
    k_low = potassium < 50

    if n_low and p_low and k_low:
        return "17-17-17"
    elif n_low and p_low:
        return "DAP"
    elif n_low and k_low:
        return "20-20"
    elif p_low and k_low:
        return "14-35-14"
    elif n_low:
        return "Urea"
    elif p_low:
        return "DAP"
    elif k_low:
        return "MOP"
    elif crop_type in ["rice", "sugarcane", "vegetables"]:
        return "28-28"
    elif crop_type in ["pulses", "fruits"]:
        return "10-26-26"
    else:
        return "17-17-17"


def generate_fertilizer_dataset(n_samples: int = 1500) -> pd.DataFrame:
    """Generate synthetic fertilizer recommendation dataset."""
    np.random.seed(42)
    records = []

    for _ in range(n_samples):
        temp = np.random.uniform(10, 45)
        humidity = np.random.uniform(20, 95)
        moisture = np.random.uniform(10, 60)
        soil_type = np.random.choice(SOIL_TYPES)
        crop_type = np.random.choice(CROP_TYPES)
        N = np.random.uniform(0, 140)
        K = np.random.uniform(0, 205)
        P = np.random.uniform(0, 145)

        fertilizer = assign_fertilizer(N, P, K, crop_type, soil_type)

        records.append({
            "Temperature": temp, "Humidity": humidity, "Moisture": moisture,
            "Soil Type": SOIL_TYPES.index(soil_type),
            "Crop Type": CROP_TYPES.index(crop_type),
            "Nitrogen": N, "Potassium": K, "Phosphorous": P,
            "Fertilizer Name": fertilizer,
        })

    return pd.DataFrame(records)


def train_fertilizer_model():
    print("\n[Step 2] Training Fertilizer Recommendation Model (Gradient Boosting)")
    print("-" * 50)

    df = generate_fertilizer_dataset(1500)
    print(f"   Dataset size: {len(df)} samples, {df['Fertilizer Name'].nunique()} fertilizers")

    feature_cols = ["Temperature", "Humidity", "Moisture", "Soil Type",
                    "Crop Type", "Nitrogen", "Potassium", "Phosphorous"]
    X = df[feature_cols].values
    y = df["Fertilizer Name"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   [OK] Test Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(model, MODEL_DIR / "fertilizer_model.pkl")
    joblib.dump(encoder, MODEL_DIR / "fertilizer_label_encoder.pkl")
    joblib.dump(scaler, MODEL_DIR / "fertilizer_scaler.pkl")
    print(f"   [Saved] -> backend/models/fertilizer_model.pkl")
    print(f"   [Saved] -> backend/models/fertilizer_label_encoder.pkl")
    print(f"   [Saved] -> backend/models/fertilizer_scaler.pkl")

    return accuracy


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    crop_acc = train_crop_model()
    fert_acc = train_fertilizer_model()

    print("\n" + "=" * 65)
    print("[OK] Training Complete!")
    print(f"   Crop Recommendation Model Accuracy    : {crop_acc * 100:.2f}%")
    print(f"   Fertilizer Recommendation Model Accuracy: {fert_acc * 100:.2f}%")
    print(f"   Models saved to: {MODEL_DIR}")
    print("\n[INFO] You can now start the server:")
    print("   python -m uvicorn backend.main:app --reload")
    print("=" * 65)
