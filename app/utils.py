import pickle
import pandas as pd
import os
import requests
import tempfile
import streamlit as st

# Load model dengan path yang benar
def load_model():
    model_path = os.path.join("model", "xgb_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

# ===== Helper untuk download & cache file dari URL =====
@st.cache_data(show_spinner=False)
def download_file(url, filename=None):
    if filename is None:
        filename = url.split("/")[-1]
    cache_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(cache_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(cache_path, "wb") as f:
            f.write(r.content)
    return cache_path

# ===== Load dataset dari Hugging Face =====
@st.cache_data(show_spinner=True)
def load_data_from_hf():
    hf_data_url = "https://huggingface.co/datasets/sidiq11/hydraulic_sales/resolve/main/hydraulic_sales.csv"  # ganti dengan link file csv kamu di HF
    csv_path = download_file(hf_data_url)
    return pd.read_csv(csv_path)

# Load data sekali
hydraulic = load_data_from_hf()


def build_features(df):
    df = df.copy()

    # Pastikan kolom date jadi datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.sort_values(["variant", "date"]).reset_index(drop=True)

    # Lags & rolling
    df["lag_1"] = df.groupby("variant")["units_sold"].transform(lambda s: s.shift(1))
    df["lag_3"] = df.groupby("variant")["units_sold"].transform(lambda s: s.shift(3))
    df["rolling_mean_3"] = df.groupby("variant")["units_sold"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )

    # Calendar features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["yearly_trend"] = df.groupby("year")["units_sold"].transform("mean")

    # Promo effectiveness
    df["promo_effectiveness"] = df["marketing_spend"] / df["yearly_trend"]

    # Optional: track NA
    for c in ["lag_1", "lag_3", "rolling_mean_3"]:
        df[f"{c}_was_na"] = df[c].isna()

    # Fill NA forward (per variant)
    for c in ["lag_1", "lag_3", "rolling_mean_3"]:
        df[c] = df.groupby("variant")[c].transform(lambda s: s.ffill())

    # Drop rows yang masih NaN di awal seri
    df = df.dropna(subset=["lag_1", "lag_3"]).reset_index(drop=True)

    return df