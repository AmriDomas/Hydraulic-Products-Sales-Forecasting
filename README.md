# 🛠 Hydraulic Products Monthly Sales Prediction

This repository contains an end-to-end machine learning pipeline for predicting monthly sales of hydraulic products.
It covers data preparation, model training, and deployment using FastAPI, Streamlit, and Docker Compose.

The core ML workflow was inspired by the [Hydraulic Products Monthly Sales Prediction Kaggle Notebook](https://www.kaggle.com/code/amri11/hydraulic-products-monthly-saless-prediction#Hydraulic-Products-Monthly-Saless-Prediction)

## ⚙️ Workflow

  1. Data Processing
     - Load hydraulic products sales dataset (data/)
     - Feature engineering (marketing spend, discount, competitor activity, etc.)
     - Save preprocessed features
  
  2. Model Training
      - Train regression/classification model (model/)
      - Serialize with joblib / pickle
  3. API Deployment (FastAPI)
      - REST endpoint /predict
      - Input: marketing/discount/economic indicators
      - Output: predicted monthly sales
  4. Dashboard Deployment (Streamlit)
      - Interactive visualization of predictions
      - Compare actual vs predicted sales
  
  5. Containerization (Docker + Compose)
      - FastAPI runs on port 8000
      - Streamlit runs on port 8502 (mapped to container 8501)
      - Services connected through Docker network

## 📂 Project Structure
  ```bash
    │── data/                   # Raw and processed datasets  
    │── model/                  # Trained machine learning models (serialized)  
    │── app/  
    │   ├── api/                # FastAPI service for inference (REST API)  
    │   │   ├── main.py  
    │   │   ├── requirements.txt  
    │   │   └── Dockerfile       # API container  
    │   │  
    │   ├── streamlit/          # Streamlit dashboard for visualization  
    │   │   ├── app.py  
    │   │   ├── requirements.txt  
    │   │   └── Dockerfile       # UI container  
    │   │  
    │   └── utils.py            # Shared utilities (e.g., model loader, feature builder)  
    │  
    │── docker-compose.yml       # Orchestration file (API + Streamlit)  
    │── README.md                # Project documentation  
  ```

## 🚀 Quick Start

  1️⃣ Clone repository
  ```bash
    git clone https://github.com/AmriDomas/Hydraulic-Products-Sales-Forecasting.git
    cd Hydraulic-Products-Sales-Forecasting
  ```
  
  2️⃣ Build and run services
  ```bash
    docker-compose up --build
  ```
  
  3️⃣ Access the services
    - FastAPI → http://localhost:8000/docs
    - Streamlit → http://localhost:8502

## 🐳 Useful Docker Commands

  1. Rebuild images without cache:
     ```bash
       docker-compose build --no-cache
     ```
  2. Restart services:
     ```bash
       docker-compose down && docker-compose up
     ```
  3. View logs:
     ```bash
       docker-compose logs -f
     ```
## 📊 Example FastAPI Request

  ```json
    POST /predict
  {
    "variant": "A",
    "date": "2025-01-01",
    "marketing_spend": 15000,
    "discount_percent": 5.0,
    "competitor_activity": 0.7,
    "economic_indicator": 1.2,
    "seasonality_index": 0.9
  }
 ```

Response:

```json
  {
  "prediction": 325
}
```

## 📝 Notes

 - Dependencies are managed separately for api/requirements.txt and streamlit/requirements.txt.
 - To add new features, update utils.py and rebuild.
 - Models must be placed in model/ directory before deployment.

## 📬 Contact

👤 Author: Amri Domas

🔗 [LinkedIn](https://www.linkedin.com/in/muh-amri-sidiq/)

📧 For collaboration or inquiries, please connect via LinkedIn.
