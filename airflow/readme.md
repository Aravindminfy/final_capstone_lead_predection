# sample output
<img width="1919" height="986" alt="Screenshot 2025-07-20 133621" src="https://github.com/user-attachments/assets/3c1831fa-ec4d-4353-98db-31b4aebcd5b0" />

# Airflow Folder — Detailed Documentation  
**Path:** `final_capstone/airflow`  
**Purpose:** Orchestrates the end-to-end MLOps workflow for sales conversion prediction using Apache Airflow, including data extraction, preprocessing, EDA, model training, drift detection, and dashboarding.

---

## 1. **Overview & Flow**

This folder contains all components required to run a **production-grade, automated MLOps pipeline** using Apache Airflow. The pipeline manages data ingestion, cleaning, analysis, model lifecycle, drift monitoring, and reporting.

**Main Flow:**
1. **Data Extraction:** Pulls raw data from source (e.g., PostgreSQL via pgAdmin).
2. **Preprocessing:** Cleans, encodes, and transforms data for ML.
3. **EDA:** Generates exploratory data analysis reports.
4. **Model Training & Scoring:** Trains models, selects best, and saves artifacts.
5. **Drift Detection:** Monitors for data drift and triggers retraining if needed.
6. **Dashboarding:** Updates Streamlit dashboard with latest results.
7. **Logging & Monitoring:** Tracks pipeline status, errors, and drift events.

---

## 2. **Key Files & Their Roles**

### **Configuration & Environment**
- `.env`: Stores environment variables (DB credentials, paths, etc.).
- `docker-compose.yml`: Defines multi-container setup for Airflow, database, and supporting services.
- `Dockerfile`: Customizes Airflow image (Python packages, dependencies).

### **DAGs (Directed Acyclic Graphs)**
Located in `dags/`, each Python file represents a pipeline step or a full workflow:

- **`extract_pgadmin.py`**  
  Extracts raw data from PostgreSQL/pgAdmin and saves to `data/`.
  - *Reason:* Ensures fresh, consistent data for downstream tasks.

- **`preprocess.py`**  
  Cleans, encodes, and transforms raw data using a reproducible pipeline.
  - *Reason:* Guarantees consistency between training and inference.

- **`eda.py`**  
  Runs automated exploratory data analysis, saves visualizations and metrics.
  - *Reason:* Provides insights for feature engineering and business understanding.

- **`mlops_full_processs.py` / `main_full_processs.py`**  
  Orchestrates the complete ML workflow: extraction, preprocessing, training, evaluation, and artifact saving.
  - *Reason:* Automates end-to-end ML lifecycle for reliability and scalability.

- **`conplete_mlops.py`**  
  (Likely a typo for `complete_mlops.py`) Runs the full MLOps pipeline as a single DAG.
  - *Reason:* Simplifies scheduling and monitoring of the entire process.

- **`modal.py`**  
  Handles model training, selection, and saving.
  - *Reason:* Modularizes model logic for reuse and maintainability.

- **`data_drift_check.PY`**  
  Monitors for data drift using statistical tests, logs drift status, and triggers retraining if drift is detected.
  - *Reason:* Maintains model accuracy and reliability in changing environments.

- **`streamlit_dash.py`**  
  Updates or triggers the Streamlit dashboard to reflect latest data and model results.
  - *Reason:* Ensures business users have access to up-to-date insights.

---

### **Data & Artifacts (`data/`)**
- **`Lead Scoring.csv` / `check.csv`**: Raw and test datasets.
- **`preprocessed_data.csv`**: Output from preprocessing step.
- **`preprocessing_pipeline.pkl`**: Saved pipeline for consistent transformation.
- **`processing_metadata.pkl`**: Feature types and metadata for reproducibility.
- **`best_model_xgboost.pkl`**: Trained model artifact.
- **`drift_detected.txt`, `drift_log.txt`, `drift_status.flag`**: Drift monitoring outputs.
- **`streamlit.log`**: Dashboard logs.

---

### **Logs (`logs/`)**
- Organized by DAG ID, contains execution logs for each pipeline run.
- *Importance:* Supports audit, debugging, and compliance.

---

## 3. **Implementation Details & Flow**

### **Airflow DAGs**
- Each DAG defines a sequence of tasks (operators) with dependencies.
- Tasks are Python functions or scripts (e.g., BashOperator, PythonOperator).
- DAGs are scheduled (e.g., daily, hourly) and can be triggered manually or by events (e.g., drift detection).

### **Pipeline Steps**
1. **Extraction:**  
   - Connects to DB, runs SQL, saves CSV.
   - *Implementation:* Uses PythonOperator, connection from `.env`.

2. **Preprocessing:**  
   - Loads raw data, applies cleaning pipeline, saves processed data and pipeline.
   - *Implementation:* Ensures same logic as training, saves artifacts for inference.

3. **EDA:**  
   - Generates plots, stats, and saves to disk.
   - *Implementation:* Uses pandas, matplotlib, seaborn, plotly.

4. **Model Training:**  
   - Trains multiple models, tunes hyperparameters, selects best, saves model.
   - *Implementation:* Uses scikit-learn, XGBoost, MLflow for tracking.

5. **Drift Detection:**  
   - Compares new data to training data using statistical tests (e.g., KS test, PSI).
   - Logs drift status and triggers retraining if needed.
   - *Implementation:* Writes to `drift_detected.txt` and sets flags.

6. **Dashboard Update:**  
   - Runs or refreshes Streamlit dashboard with latest results.
   - *Implementation:* May use BashOperator to run Streamlit or update files.

7. **Logging & Monitoring:**  
   - All steps log to Airflow and local files.
   - *Implementation:* Uses Airflow logging and custom log files.

---

## 4. **Production-Level Importance**

- **Automation:**  
  Airflow schedules and monitors all steps, reducing manual intervention.
- **Reproducibility:**  
  Pipelines, models, and metadata are saved and versioned.
- **Scalability:**  
  Docker and Airflow support distributed, parallel execution.
- **Auditability:**  
  Logs and artifacts enable traceability and compliance.
- **Reliability:**  
  Drift detection and retraining maintain model performance.
- **Business Alignment:**  
  Dashboard and reporting keep stakeholders informed.

---

## 5. **How to Use**

1. **Configure `.env`** with DB credentials, paths, and secrets.
2. **Build Docker containers** using `docker-compose up`.
3. **Access Airflow UI** to monitor and trigger DAGs.
4. **Review logs and data artifacts** in `data/` and `logs/`.
5. **Check Streamlit dashboard** for latest business insights.
6. **Monitor drift status** and retraining events.

---

## 6. **Extensibility**

- Add new DAGs for additional ML tasks (e.g., model monitoring, explainability).
- Integrate with cloud storage (S3, Azure Blob) for data and artifacts.
- Connect to CI/CD for automated deployment.
- Enhance drift detection with advanced statistical or ML methods.
- Schedule retraining and reporting based on business needs.

---

## 7. **Summary Table**

| Component         | Purpose                                  | Implementation/Reason                |
|-------------------|------------------------------------------|--------------------------------------|
| `.env`            | Store secrets/config                     | Secure, flexible configuration       |
| `docker-compose.yml` | Multi-container orchestration         | Scalable, reproducible environment   |
| `dags/`           | Pipeline steps as DAGs                   | Modular, maintainable workflow       |
| `data/`           | Data and model artifacts                 | Versioned, auditable outputs         |
| `logs/`           | Execution logs                           | Debugging, compliance                |
| Drift Detection   | Monitor data changes                     | Maintain model reliability           |
| Dashboard         | Business insights                        | Stakeholder communication            |

---

## 8. **Conclusion**

The Airflow folder provides a **complete, automated, and production-ready MLOps orchestration** for sales conversion prediction. It ensures data integrity, model reliability, and business alignment through modular DAGs, robust logging, and automated drift monitoring.

---


