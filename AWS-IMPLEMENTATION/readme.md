<img width="1917" height="951" alt="image" src="https://github.com/user-attachments/assets/9d695f4b-426f-49f3-b6d1-9bcac9b4de6c" />
<img width="912" height="1275" alt="complete_end_to_end_aws_mlops" src="https://github.com/user-attachments/assets/b23895b3-6a69-45a2-ae02-4b56b9bf2a9c" />
![final_porject_articture](https://github.com/user-attachments/assets/14888ccf-9e31-453a-852d-04f18108a28e)

# Production ML System:  Data Pipeline with Automated Drift Detection & Manual Retraining Approval

## Architecture Overview

This production-ready machine learning system implements a comprehensive MLOps pipeline designed for enterprise-scale lead conversion prediction. The architecture follows best practices for data governance, automated monitoring, and human-in-the-loop decision making.

## System Components

### 1. Data Ingestion Layer

**Flask REST API**
- **Purpose**: Primary entry point for real-time data ingestion
- **Features**:
  - FastAPI/Gunicorn for high-performance request handling
  - Pydantic input validation for data quality assurance
  - Rate limiting and authentication for security
  - MySQL tunnel for secure database connections
  - JSON payload processing for standardized data format
- **Data Flow**: Client → POST /api/v1/predict → JSON Payload → Processing Pipeline

### 2. Data Storage & Processing

**RedShift Data Warehouse**
- **Components**:
  - Columnar storage for analytical workloads
  - Materialized views for optimized query performance
  - Query optimization and connection pooling
  - Analytics workbench for data exploration
- **Purpose**: Central repository for all training and inference data

**AWS Glue ETL Pipeline**
- **Spark Jobs**: Distributed data processing using PySpark
- **Data Transformation**: Clean, validate, and structure incoming data
- **Data Quality Checks**: Automated validation rules and anomaly detection
- **Feature Engineering**: Create derived features for model training
- **Incremental Processing**: Process only new/changed data for efficiency

**Lambda Triggers**
- **Event-Driven**: Automatically triggered by new data arrivals
- **Real-time Processing**: Immediate data transformation and validation
- **Scalable**: Auto-scaling based on data volume

**AWS S3 Data Lake**
- **Partitioned Data Format**: Organized by date/source for efficient querying
- **Lifecycle Policies**: Automated data archival and retention
- **Event Notifications**: Trigger downstream processes
- **Data Versioning**: Track data lineage and changes

### 3. ML Training & Registry

**MLflow Model Registry**
- **Model Versioning**: Track all model versions and experiments
- **A/B Testing Config**: Support for model comparison testing
- **Performance Metrics**: Store and compare model performance
- **Model Lineage**: Track data and code used for each model
- **Deployment History**: Record of all model deployments

**SageMaker Training**
- **Distributed Training**: Scale training across multiple instances
- **Hyperparameter Tuning**: Automated parameter optimization
- **Model Artifacts (S3)**: Store trained models and metadata
- **Training Metrics**: Monitor training progress and performance

### 4. Automated Drift Detection

**Evidently Drift Monitor**
- **Statistical Tests**: Kolmogorov-Smirnov (KS) and Population Stability Index (PSI)
- **Feature Drift Detection**: Monitor individual feature distributions
- **Data Quality Monitoring**: Track missing values, outliers, and data types
- **Performance Monitoring**: Compare current vs. baseline model performance
- **Dashboard Updates**: Real-time visualization of drift metrics
- **Email Notifications**: Alert stakeholders when drift is detected
- **No Auto-Retraining**: Prevents automatic model updates without approval

### 5. Manual Approval Process

**Human-in-the-Loop Workflow**
- **Data Scientist Review Required**: Expert validation before retraining
- **Business Impact Assessment**: Evaluate potential business consequences
- **Resource Planning & Scheduling**: Allocate computational resources
- **Data Analysis Validation**: Verify data quality and drift analysis
- **Testing Environment**: Safe space for model validation
- **Conscious Retrain Decision**: Deliberate choice to proceed
- **Audit Trail & Compliance**: Full documentation for regulatory requirements

### 6. Inference & Serving

**Flask Inference API**
- **Model Loading**: Load trained models from MLflow registry
- **Real-time Predictions**: Low-latency prediction serving
- **Response Caching**: Cache frequent predictions for performance
- **Load Balancing**: Distribute traffic across multiple instances
- **Monitoring Hooks**: Capture prediction requests and responses

### 7. Automated Monitoring

**MLOps Dashboard**
- **Drift Analysis Reports**: Comprehensive drift detection summaries
- **Model Performance Metrics**: Track accuracy, precision, recall over time
- **Retrain Recommendations**: AI-powered suggestions for retraining
- **Manual Trigger Controls**: Allow operators to initiate processes
- **Approval Tracking**: Monitor approval workflow status

## Data Flow Process

### Training Pipeline
1. **Data Ingestion**: Raw data enters via Flask REST API
2. **Storage**: Data stored in RedShift warehouse and S3 data lake
3. **Processing**: AWS Glue ETL transforms and validates data
4. **Training**: SageMaker trains models with processed data
5. **Registry**: Models registered in MLflow with metadata
6. **Deployment**: Approved models deployed to inference API

### Inference Pipeline
1. **Client Request**: Real-time prediction request via Flask API
2. **Model Loading**: Current production model loaded from registry
3. **Prediction**: Model generates conversion probability
4. **Response**: JSON response with prediction and confidence
5. **Monitoring**: Request/response logged for drift detection

### Drift Detection & Retraining Workflow
1. **Continuous Monitoring**: Evidently monitors data and model performance
2. **Drift Detection**: Statistical tests identify significant changes
3. **Alert Generation**: Notifications sent to data science team
4. **Manual Review**: Data scientists assess drift significance
5. **Approval Decision**: Human approval required for retraining
6. **Retraining**: If approved, new model training initiated
7. **Validation**: New model tested before deployment
8. **Deployment**: Approved model replaces current production model

## Key Features

### Governance & Compliance
- **Human Oversight**: Manual approval prevents unwanted model changes
- **Audit Trail**: Complete logging of all decisions and changes
- **Data Lineage**: Track data from source to prediction
- **Version Control**: All models, data, and code versioned

### Scalability & Performance
- **Horizontal Scaling**: Components scale independently
- **Caching**: Multiple levels of caching for performance
- **Load Balancing**: Traffic distributed across instances
- **Resource Optimization**: Efficient use of computational resources

### Reliability & Monitoring
- **Automated Testing**: Continuous validation of data and models
- **Health Checks**: Monitor system component status
- **Rollback Capability**: Quick reversion to previous model versions
- **Alert System**: Proactive notification of issues

### Security
- **Authentication**: Secure API access control
- **Rate Limiting**: Prevent abuse and ensure fair usage
- **Data Encryption**: Protect sensitive data in transit and at rest
- **Access Control**: Role-based permissions for different operations

## Technology Stack

- **APIs**: Flask, FastAPI, Gunicorn
- **Data Storage**: AWS RedShift, S3 Data Lake
- **Data Processing**: AWS Glue (Spark/PySpark), Lambda
- **ML Training**: Amazon SageMaker
- **Model Registry**: MLflow
- **Monitoring**: Evidently AI
- **Orchestration**: Airflow (implied for scheduling)
- **Infrastructure**: AWS Cloud Services

This architecture ensures production-grade reliability, scalability, and governance for machine learning operations while maintaining the flexibility to adapt to changing business requirements.
