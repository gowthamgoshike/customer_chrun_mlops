# Customer Churn Prediction MLOps 

## Overview

This project implements a **production-ready end-to-end machine learning system** for predicting customer churn.

The system includes:

- Data processing pipelines
- Feature engineering
- Machine learning model training
- Experiment tracking
- Model API deployment
- Interactive analytics dashboard
- Docker containerization
- Cloud deployment

The goal is to simulate a **real-world ML system used in production environments**.

---

# Architecture

```
Raw Data
   ↓
Data Processing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
MLflow Experiment Tracking
   ↓
Model Registry
   ↓
FastAPI Prediction API
   ↓
Docker Container
   ↓
Streamlit Dashboard
   ↓
User Predictions
```

---

# Project Structure

```
customer_churn_mlops
│
├── api
│   └── main.py
│
├── dashboard
│   └── app.py
│
├── docker
│   └── Dockerfile
│
├── src
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── data
│   ├── raw
│   └── processed
│
├── models
│
├── notebooks
│
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python
- Scikit-Learn
- MLflow
- FastAPI
- Streamlit
- Docker
- Git
- AWS EC2

---

# Features

- End-to-end machine learning pipeline
- Experiment tracking using MLflow
- REST API for predictions
- Interactive web dashboard
- Containerized deployment
- Cloud-ready architecture

---

# Installation

Clone repository

```
git clone https://github.com/gowthamgoshike/customer_churn_mlops_system.git
cd customer_churn_mlops_system
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Run Training Pipeline

```
python src/train_model.py
```

---

# Run FastAPI Service

```
uvicorn api.main:app --reload
```

API documentation:

```
http://127.0.0.1:8000/docs
```

---

# Run Dashboard

```
streamlit run dashboard/app.py
```

Dashboard:

```
http://localhost:8501
```

---

# Docker Deployment

Build container:

```
docker build -t churn-ml-api -f docker/Dockerfile .
```

Run container:

```
docker run -p 8000:8000 churn-ml-api
```

---

# Future Improvements

- CI/CD pipeline with GitHub Actions
- Kubernetes deployment
- Feature store integration
- Automated model retraining

---

# Author

Gowtham Goshike

Full Stack Data Scientist | Machine Learning Engineer
