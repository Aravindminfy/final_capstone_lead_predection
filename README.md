<img width="779" height="585" alt="local" src="https://github.com/user-attachments/assets/453fbb1a-a948-49d6-a6f9-614affb49288" />

# Local Production ML Pipeline Architecture

This repository implements a complete **end-to-end machine learning pipeline** for **on-premises (local) deployment**. The architecture emphasizes modularity, real-time prediction, model drift detection, and human-in-the-loop retraining, making it suitable for environments with strict data privacy and sovereignty requirements.

## Architecture Overview

The pipeline follows a horizontal data flow from client data submission through to model retraining approval.

### Primary Data Flow

1. **Data Ingestion**
   - Client applications send data to a **Flask REST API**.
   - Incoming raw data is stored in **MinIO** (a local, S3-compatible object storage service).

2. **ETL Pipeline (Airflow)**
   - **Apache Airflow** orchestrates daily ETL processes:
     - Data Cleaning
     - Feature Engineering
   - Processed data is loaded into a **PostgreSQL** database (acting as the local data warehouse).

3. **Model Training**
   - Cleaned and feature-enriched data is used to train machine learning models.
   - Trained models are logged and registered with **MLflow Model Registry** for version control.

4. **Model Deployment**
   - The latest approved model is deployed to the **Flask Prediction API** for real-time inference.

### Drift Detection and Monitoring Loop

1. **Daily Drift Monitoring**
   - A **Drift Monitor** runs as a scheduled Airflow task.
   - It compares incoming data distributions with the training data using statistical checks (e.g., Kolmogorov-Smirnov test).
   - A drift is flagged when the divergence exceeds a **0.15 threshold**.

2. **Alert System**
   - Upon drift detection, an alert is automatically sent via **email or Slack**.
   - Alerts include summary metrics and visual reports from tools like **Evidently AI**.

3. **Approval Workflow**
   - Data scientists are directed to an **Approval Dashboard** to:
     - Review drift reports.
     - Validate alert accuracy.
     - Approve or reject model retraining.

4. **Retraining**
   - If approved, a new model training pipeline is triggered.
   - The updated model is again registered in MLflow and redeployed to the API after evaluation.

## Key Features

- **Local Data Control**: Full data sovereignty with no cloud dependencies.
- **Modular Design**: Separates ingestion, processing, training, and deployment components.
- **Real-Time Predictions**: Models served via a Flask API for low-latency inference.
- **Automated Drift Detection**: Statistical monitoring of input data quality.
- **Human-in-the-Loop Retraining**: Data scientists make retraining decisions through an approval dashboard.
- **Auditability**: All model versions and metrics are tracked via MLflow.

## Technology Stack

| Component          | Tool               |
|--------------------|--------------------|
| Data Ingestion      | Flask API           |
| Object Storage      | MinIO (S3-compatible)|
| Workflow Orchestration | Apache Airflow   |
| Data Warehouse      | PostgreSQL         |
| Model Training      | Python (Scikit-learn / XGBoost / etc.) |
| Model Registry      | MLflow              |
| Monitoring & Alerts | Evidently AI, Slack/Email |
| Dashboard Interface | Streamlit / Flask UI |

## Deployment Notes

- All services can be containerized via Docker.
- Airflow and MLflow can be configured to run in local Docker environments.
- System can be extended to support CI/CD via GitHub Actions or GitLab Pipelines.

## Recommended Folder Structure





## Project Structure

├── api/ # Flask prediction and data ingestion APIs\
├── airflow/ # DAGs and Airflow configuration\
├── data/ # Processed and raw data (MinIO mounted)\
├── drift_monitor/ # Drift detection scripts and reports\
├── mlflow/ # MLflow setup and model tracking\
├── model_training/ # Scripts for training and evaluation\
├── postgres/ # PostgreSQL setup for metadata and features\
├── dashboard/ # Approval UI ( Flask)\
├── requirements.txt # Python dependencies\
└── README.md # This file\


## Tools and Technologies

- Python 3.8+
- Apache Airflow
- MLflow
- Flask
- Streamlit
- Evidently AI
- Docker
- AWS CLI (S3/EC2)
- PostgreSQL (optional, if used for metadata storage)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AWS-IMPLEMENTATION.git
cd AWS-IMPLEMENTATION
```
### 2 . Create Virtual Environment and Install Dependencies
```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
### 3. Load Raw Data
Place your raw CSV or data files into the raw_data/ directory. You can also configure your scripts to fetch data directly from AWS S3 if needed.
### 4.Perform Exploratory Data Analysis (EDA)
```
cd streamlit_dashboard_EDA
streamlit run app.py
```
Open the browser link shown in the terminal to explore your data visually.

### 4. Run Data Preprocessing
```
cd preprocess
python preprocess_leads.py
```
### 5.Train and Evaluate the Model
```
cd modal_development
python train_model.py
```
This script will train the lead scoring model and save it for tracking and deployment.
### 6. Track Experiments Using MLflow
```
cd mlflow
mlflow ui
```
Navigate to http://localhost:5000 in your browser to view and compare model runs, metrics, and parameters.
### 7.  Serve the Model Using Flask
```
cd flask_lead_scoring
python app.py
```
The model will be available via a REST API at 

```http://localhost:5000/predict.```

### 10. Monitor Model Using Evidently:
```
cd evidently
python monitor.py
```
Visit the Airflow UI at ```http://localhost:8080``` Log in and enable your DAGs to start the orchestration.\

Note: Set Airflow variables and connections for AWS S3, PostgreSQL, or MLflow inside the Airflow UI under Admin > Variables/Connections.



