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
