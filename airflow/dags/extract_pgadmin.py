# extract_lead_data.py
"""
Extracts data from PostgreSQL and saves it as lead_scoring.csv.
"""

import pandas as pd
import psycopg2
import os

def main():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "aravind"),
        user=os.getenv("DB_USER", "aravind"),
        password=os.getenv("DB_PASS", "123"),
        port=os.getenv("DB_PORT", "5432")
    )
    query = "SELECT * FROM lead_data"
    df = pd.read_sql(query, conn)
    conn.close()
    
    output_path = os.path.join(os.getenv("DATA_DIR", "data"), "Lead Scoring.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Data extracted and saved to {output_path}")

if __name__ == "__main__":
    main()
