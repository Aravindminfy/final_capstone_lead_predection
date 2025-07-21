<<<<<<< HEAD
FROM apache/airflow:2.5.3-python3.10

# Install system packages
USER root
RUN apt-get update && \
    apt-get install -y gcc g++ build-essential && \
    rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy and install Python dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
=======
FROM apache/airflow:2.7.1-python3.9

USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --upgrade pip setuptools wheel

# Install all your needed Python packages here
RUN pip install \
    pandas \
    numpy \
    scikit-learn \
    psycopg2-binary \
    pyyaml \
    mlflow \
    xgboost \
    joblib \
    evidently

RUN pip install flask-session apache-airflow-providers-postgres apache-airflow-providers-google apache-airflow-providers-amazon

RUN pip install pandas numpy scikit-learn psycopg2-binary pyyaml mlflow xgboost joblib evidently
COPY ./dags /opt/airflow/dags
COPY ./plugins /opt/airflow/plugins
>>>>>>> 5c6ff1f (18-7-2025 commit)
