import streamlit as st

# ===== MUST be the first Streamlit command =====
st.set_page_config(page_title="Hydraulic Product Sales App", layout="wide")
# =================================================

import sys
import os
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import requests
import tempfile
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import numpy as np


# Tambahkan path ke sys.path
current_dir = Path(__file__).parent
app_dir = current_dir.parent
sys.path.append(str(app_dir))

from utils import load_model, build_features

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

# ====== Load dataset ======
# Load data dan model
hydraulic = load_data_from_hf()
cat_imputer = SimpleImputer(strategy="most_frequent")
hydraulic['product_code'] = cat_imputer.fit_transform(hydraulic[['product_code']]).ravel()

num_imputer = SimpleImputer(strategy="median")
hydraulic['revenue'] = num_imputer.fit_transform(hydraulic[['revenue']])

st.title("📈 Hydraulic Products Sales Forecasting")
st.write("Forecast monthly sales using XGBoost + Time Series Features")

# ====== Sidebar & Navigation ======
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select Menu", ["Analysis", "Prediction"])

# ====== ANALYSIS ======
if menu == "Analysis":
    st.title("📊 Data Analysis - Hydraulic Product Sales")
    df = hydraulic.copy()

    # CSS untuk tab biar rata
    st.markdown("""
    <style>
    /* Tab container rata */
    div[data-baseweb="tab-list"] {
        justify-content: space-between !important;
    }

    /* Ukuran font tab 10px */
    div[data-baseweb="tab"] > button {
        font-size: 20px !important;
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 4px;
        padding: 6px 10px !important;
        line-height: 1.2 !important;
    }

    /* Tab aktif */
    div[data-baseweb="tab"][aria-selected="true"] > button {
        background-color: #1976d2 !important;
        color: white !important;
        border-radius: 6px 6px 0 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tab setup
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Preview", "📈 Visualization", "🔥 Correlation", "🎨 Custom Plot", "📈 Time Series By Month"
    ])

    def get_dynamic_palette(n):
            """Generate palette: if n>3 use darkening gradient, else fixed Set2."""
            if n <= 3:
                return sns.color_palette("Set2", n)
            else:
                # gradasi dari terang ke gelap (Blues)
                return sns.color_palette("Blues", n)

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Basic Statistics")
        st.write(df.describe(include='all'), use_container_width=True)
    
    def plot_categorical(df, col):
        num_cat = df[col].nunique(dropna=False)
        palette = get_dynamic_palette(num_cat)

        fig, ax = plt.subplots(figsize=(6, 3))
        order = df[col].value_counts().index
        sns.countplot(x=col, data=df, order=order, palette=palette, ax=ax)

        # Title & ticks lebih proporsional
        ax.set_title(f"Distribusi {col}", fontsize=8, weight='bold')
        ax.set_xlabel(col, fontsize=6)  # benerin param
        ax.set_ylabel("Count", fontsize=6)
        ax.tick_params(axis='x', rotation=45, labelsize=5)
        ax.tick_params(axis='y', labelsize=5)

        # Label bar lebih kecil, posisinya di edge
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=4, label_type='edge', padding=1)

        sns.despine()  # buang border luar
        st.pyplot(fig)


    with tab2:

        st.markdown("""
        <style>
        [data-testid="stDataFrame"] {
            width: 100% !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.subheader("📊 Distribution Category")
        # exclude hanya kolom 'Id' dan 'Timestamp' (persis nama itu)
        exclude_cols = ["date"]

        kategori_cols = [c for c in df.select_dtypes(include='object').columns if c not in exclude_cols]

        col_kategori = st.selectbox("Select the category column", kategori_cols)

        col1, col2 = st.columns([1,1])

        with col1:
            st.subheader("📊 Distribusi Kategori (Bar Chart)")
            fig, ax = plt.subplots(figsize=(5, 4))  

            order = df[col_kategori].value_counts().index
            num_cat = len(order)
            palette = get_dynamic_palette(num_cat)

            sns.countplot(
                x=col_kategori, 
                data=df, 
                order=order, 
                palette=palette, 
                width=0.6,  # lebih ramping
                ax=ax
            )

            ax.set_title(f"Distribution {col_kategori}", fontsize=10, weight='bold')
            ax.set_xlabel(col_kategori, fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.tick_params(axis='y', labelsize=7)

            for container in ax.containers:
                ax.bar_label(container, fmt='%d', fontsize=6, label_type='edge', padding=1)

            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Distribusi Kategori (Pie Chart)")
            val_counts = df[col_kategori].value_counts()

            # ukuran fix biar gak berubah-ubah
            fig, ax = plt.subplots(figsize=(5, 4))  

            wedges, texts, autotexts = ax.pie(
                val_counts,
                labels=None,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette("Blues", len(val_counts))
            )

            # kunci aspect ratio -> pie selalu lingkaran sempurna
            ax.axis('equal')

            percentages = val_counts / val_counts.sum() * 100
            ax.legend(
                wedges,
                [f"{cat} ({p:.1f}%)" for cat, p in zip(val_counts.index, percentages)],
                title=col_kategori,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=6,
                title_fontsize=8
            )
            
            plt.tight_layout()
            st.pyplot(fig)


    with tab3:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df.drop(columns=["id"], errors='ignore').corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="Spectral", ax=ax)
        st.pyplot(fig)

    with tab4:
        st.subheader("📊 Custom Plot (Auto Bar / Scatter / Box)")
        all_cols = [c for c in df.columns if c.lower() != 'date']
        col_x = st.selectbox("Select Column X", all_cols, index=0)
        col_y = st.selectbox("Select Column Y", all_cols, index=1)

        fig, ax = plt.subplots(figsize=(8, 4))
        x_is_cat = df[col_x].dtype == 'object'
        y_is_cat = df[col_y].dtype == 'object'

        if x_is_cat and y_is_cat:
            num_cat = df[col_y].nunique(dropna=False)
            palette = get_dynamic_palette(num_cat)
            sns.countplot(x=col_x, hue=col_y, data=df, palette=palette, ax=ax)
            ax.set_title(f"Bar Chart: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.set_xlabel(col_x, fontsize=6)
            ax.set_ylabel("Count", fontsize=6)
            ax.tick_params(axis='x', rotation=45, labelsize=5)
            ax.tick_params(axis='y', labelsize=5)
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', fontsize=4, label_type='edge', padding=1)

        elif not x_is_cat and not y_is_cat:
            sns.scatterplot(x=col_x, y=col_y, data=df, color="royalblue", ax=ax)
            ax.set_title(f"Scatter Plot: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.set_xlabel(col_x, fontsize=6)
            ax.set_ylabel(col_y, fontsize=6)
            ax.tick_params(axis='both', labelsize=5)

        else:
            if x_is_cat:
                num_cat = df[col_x].nunique(dropna=False)
                palette = get_dynamic_palette(num_cat)
                sns.boxplot(x=col_x, y=col_y, data=df, palette=palette, ax=ax)
                ax.set_xlabel(col_x, fontsize=6)
                ax.set_ylabel(col_y, fontsize=6)
            else:
                num_cat = df[col_y].nunique(dropna=False)
                palette = get_dynamic_palette(num_cat)
                sns.boxplot(x=col_y, y=col_x, data=df, palette=palette, ax=ax)
                ax.set_xlabel(col_y, fontsize=6)
                ax.set_ylabel(col_x, fontsize=6)

            ax.set_title(f"Box Plot: {col_x} vs {col_y}", fontsize=8, weight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=5)
            ax.tick_params(axis='y', labelsize=5)

        # Atur legend biar kecil
        leg = ax.get_legend()
        if leg:
            leg.set_title(leg.get_title().get_text(), prop={'size': 5})
            for text in leg.get_texts():
                text.set_fontsize(5)

        sns.despine()
        st.pyplot(fig)

    with tab5:
        st.subheader("📈 Time Series Analysis")

        month_col = "date"

        # Pilih group column
        group_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c != month_col]
        group_col = st.selectbox("Select Group Column (Optional)", ["(None)"] + group_cols)

        # Pilih kolom numerik (multi-select untuk stacked area)
        num_cols = df.select_dtypes(include='number').columns.tolist()
        value_cols = st.multiselect("Select Numeric Column(s)", num_cols, default=[num_cols[0]])

        # Pilih metode agregasi
        agg_method = st.selectbox("Aggregation Method", ["sum", "mean", "median"])

        # Pilih jenis plot
        plot_type = st.radio("Select Plot Type", ["Line Plot", "Stacked Area Plot"])

        # Pastikan date dalam datetime
        df[month_col] = pd.to_datetime(df[month_col], errors='coerce')

        # ==== AGGREGATION ====
        if group_col != "(None)":
            df_month = df.groupby([month_col, group_col])[value_cols].agg(agg_method).reset_index()
        else:
            df_month = df.groupby(month_col)[value_cols].agg(agg_method).reset_index()

        # ==== PLOT ====
        fig, ax = plt.subplots(figsize=(9, 5))

        if plot_type == "Line Plot":
            # Kalau hanya satu kolom numerik
            if len(value_cols) == 1:
                if group_col != "(None)":
                    sns.lineplot(
                        x=month_col, y=value_cols[0],
                        hue=group_col, data=df_month,
                        marker="o", ax=ax
                    )
                else:
                    sns.lineplot(
                        x=month_col, y=value_cols[0],
                        data=df_month, marker="o",
                        color="royalblue", ax=ax
                    )

                # Label angka kalau single series
                if group_col == "(None)":
                    for x, y in zip(df_month[month_col], df_month[value_cols[0]]):
                        ax.text(x, y, f"{y:,.0f}", fontsize=6, ha='center', va='bottom')

                ax.set_ylabel(f"{value_cols[0]} ({agg_method})", fontsize=8)

            else:
                # Multi-line untuk banyak numeric columns
                for col in value_cols:
                    sns.lineplot(x=month_col, y=col, data=df_month, marker="o", ax=ax, label=col)

                ax.set_ylabel(f"Values ({agg_method})", fontsize=8)

            ax.set_title(f"Trend by Month ({agg_method})", fontsize=10, weight='bold')

        else:  # ==== STACKED AREA ====
            if group_col != "(None)":
                # Pivot: row=month, col=group, values=value_cols[0] (hanya 1 kolom numerik)
                if len(value_cols) == 1:
                    pivot_df = df_month.pivot(index=month_col, columns=group_col, values=value_cols[0]).fillna(0)
                    pivot_df.plot.area(ax=ax, stacked=True, colormap="tab20")
                    ax.set_ylabel(f"{value_cols[0]} ({agg_method})", fontsize=8)
                else:
                    st.warning("Stacked Area with groups only supports 1 numeric column. Select only 1.")
            else:
                # Stacked area untuk multi-numeric
                pivot_df = df_month.set_index(month_col)[value_cols]
                pivot_df.plot.area(ax=ax, stacked=True, colormap="tab20")
                ax.set_ylabel(f"Values ({agg_method})", fontsize=8)

            ax.set_title(f"Stacked Area by Month ({agg_method})", fontsize=10, weight='bold')

        # Format sumbu
        ax.set_xlabel("Month", fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        sns.despine()
        plt.tight_layout()
        st.pyplot(fig)

# ====== PREDICTION ======
elif menu == "Prediction":
    st.title("Hydraulic Product Sales Prediction")

    # CSS untuk tab biar rata
    st.markdown("""
    <style>
    /* Tab container rata */
    div[data-baseweb="tab-list"] {
        justify-content: space-between !important;
    }

    /* Ukuran font tab 10px */
    div[data-baseweb="tab"] > button {
        font-size: 20px !important;
        text-align: center !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 4px;
        padding: 6px 10px !important;
        line-height: 1.2 !important;
    }

    /* Tab aktif */
    div[data-baseweb="tab"][aria-selected="true"] > button {
        background-color: #1976d2 !important;
        color: white !important;
        border-radius: 6px 6px 0 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tab setup
    T1, T2, T3 = st.tabs([
        "📈 Sales Prediction Form", "📊 Feature Importance", "Marketing ROI Analysis"
    ])

    with T1:

        # Load model
        model = load_model()

        hydraulic_feat = build_features(hydraulic)

        # 2. Split train/test
        train = hydraulic_feat[hydraulic_feat["date"] < "2024-01"].copy()
        test  = hydraulic_feat[hydraulic_feat["date"] >= "2024-01"].copy()

        # 3. Feature list
        features = [
            "lag_1", "lag_3", "rolling_mean_3",
            "marketing_spend", "discount_percent",
            "competitor_activity", "economic_indicator",
            "seasonality_index", "month", "yearly_trend",
            "promo_effectiveness"
        ]

        X_train, y_train = train[features], train["units_sold"]
        X_test,  y_test  = test[features],  test["units_sold"]

        # 4. Predict
        y_pred = model.predict(X_test)

        # Gabungkan test data dengan prediksi
        plot_df = test.copy()
        plot_df["y_pred"] = y_pred

        # Plot time series
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test["date"], y_test, label="Actual", marker="o")
        ax.plot(test["date"], y_pred, label="Predicted", marker="x")
        ax.set_title("Actual vs Predicted (Test set)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Units Sold")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # tampilkan di streamlit
        st.pyplot(fig)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        smape = 100 * np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test)))

        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("sMAPE", f"{smape:.2f}%")
    
    with T2:
        st.subheader("Model Feature Importance")

        # Load model
        model = load_model()

        # Features (harus sama dengan waktu training)
        features = [
            "lag_1", "lag_3", "rolling_mean_3",
            "marketing_spend", "discount_percent",
            "competitor_activity", "economic_indicator",
            "seasonality_index", "month", "yearly_trend",
            "promo_effectiveness"
        ]

        # --- Feature Importance ---
        importance = model.feature_importances_
        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        # Barplot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(imp_df["Feature"], imp_df["Importance"], color="skyblue")
        ax.set_title("Feature Importance", fontsize=10)
        ax.set_xlabel("Importance", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)  # ukuran tulisan angka di sumbu X
        ax.tick_params(axis="y", labelsize=6)
        ax.invert_yaxis()  # biar feature terpenting di atas
        st.pyplot(fig)

        # Optional: tampilkan tabel
        st.dataframe(imp_df)
    
    with T3:
        st.subheader("Marketing ROI Analysis")

        # Agregasi marketing spend & sales
        roi_hydraulic = hydraulic.groupby("date").agg({
            "marketing_spend": "sum",
            "units_sold": "sum"
        }).reset_index()

        # Plot korelasi marketing spend vs sales
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.regplot(
            x="marketing_spend",
            y="units_sold",
            data=roi_hydraulic,
            scatter_kws={"s": 50, "alpha": 0.7},
            line_kws={"color": "red"},
            ax=ax
        )
        ax.set_title("Marketing Spend vs Sales (with correlation line)", fontsize=10)
        ax.set_xlabel("Marketing Spend", fontsize=8)
        ax.set_ylabel("Units Sold", fontsize=8)

        st.pyplot(fig)

        # Hitung korelasi
        corr = roi_hydraulic["marketing_spend"].corr(roi_hydraulic["units_sold"])
        st.metric("Correlation Marketing Spend vs Sales", f"{corr:.2f}")

    
