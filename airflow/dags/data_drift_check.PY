import os
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# === Configuration ===
# File Paths
REFERENCE_CSV = "/opt/airflow/data/lead Scoring.csv"
CURRENT_CSV = "/opt/airflow/data/check.csv"

# Get the directory where input files are located
INPUT_DIR = os.path.dirname(REFERENCE_CSV)

# Output files will be saved in the same directory as input files
DRIFT_REPORT_PATH = os.path.join(INPUT_DIR, "drift_report.html")
DRIFT_FLAG_FILE = os.path.join(INPUT_DIR, "drift_detected.txt")
LOG_FILE = os.path.join(INPUT_DIR, "drift_log.txt")
DRIFT_STATUS_FLAG = os.path.join(INPUT_DIR, "drift_status.flag")

# Drift Detection Parameters
DRIFT_THRESHOLD = 0.3  # PSI threshold for drift detection
DRIFT_METHODS = ['psi', 'ks', 'chisquare', 'wasserstein']  # Multiple drift detection methods

# === Utility Functions ===
def log_message(message, log_file=LOG_FILE):
    """Write message to log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    try:
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(log_entry)
        print(message)  # Also print to console
    except Exception as e:
        print(f"Error writing to log file: {e}")
        print(message)  # Still print to console

def load_and_validate_data(reference_path, current_path):
    """Load and validate reference and current datasets."""
    try:
        reference_df = pd.read_csv(reference_path)
        current_df = pd.read_csv(current_path)
        
        log_message(f"Reference data shape: {reference_df.shape}")
        log_message(f"Current data shape: {current_df.shape}")
        
        # Check if datasets have common columns
        common_cols = reference_df.columns.intersection(current_df.columns)
        if len(common_cols) == 0:
            raise ValueError("No common columns found between reference and current datasets")
        
        log_message(f"Common columns: {len(common_cols)} out of {len(reference_df.columns)} reference columns")
        
        return reference_df, current_df
    except Exception as e:
        log_message(f"Error loading data: {e}")
        raise

def identify_feature_types(df):
    """Automatically identify numerical and categorical features."""
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    log_message(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    log_message(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    return numerical_features, categorical_features

def log_basic_statistics(reference_df, current_df):
    """Log basic statistics for both datasets."""
    log_message("=== BASIC STATISTICS ===")
    
    # Dataset shapes
    log_message(f"Reference dataset: {reference_df.shape[0]} rows, {reference_df.shape[1]} columns")
    log_message(f"Current dataset: {current_df.shape[0]} rows, {current_df.shape[1]} columns")
    
    # Missing values
    ref_missing = reference_df.isnull().sum().sum()
    current_missing = current_df.isnull().sum().sum()
    log_message(f"Reference missing values: {ref_missing} ({ref_missing/reference_df.size*100:.2f}%)")
    log_message(f"Current missing values: {current_missing} ({current_missing/current_df.size*100:.2f}%)")
    
    # Data types comparison
    common_cols = reference_df.columns.intersection(current_df.columns)
    dtype_changes = []
    for col in common_cols:
        if str(reference_df[col].dtype) != str(current_df[col].dtype):
            dtype_changes.append(f"{col}: {reference_df[col].dtype} -> {current_df[col].dtype}")
    
    if dtype_changes:
        log_message(f"Data type changes detected: {dtype_changes}")
    else:
        log_message("No data type changes detected")

def detect_drift_with_evidently(reference_df, current_df, method='psi'):
    """
    Run Evidently data drift detection and return drift status.
    """
    log_message(f"Running drift detection using {method} method...")
    
    # Ensure common columns
    common_cols = reference_df.columns.intersection(current_df.columns)
    ref_df_clean = reference_df[common_cols].copy()
    current_df_clean = current_df[common_cols].copy()
    
    drift_detected = False
    drift_details = {}
    
    try:
        # Create drift report
        drift_report = Report([DataDriftPreset(method=method)])
        drift_report.run(reference_data=ref_df_clean, current_data=current_df_clean)
        
        # Save HTML report in the input directory
        html_path = os.path.join(INPUT_DIR, f"evidently_drift_{method}.html")
        drift_report.save_html(html_path)
        log_message(f"Drift report saved: {html_path}")
        
        # Extract drift metrics from JSON
        json_data = json.loads(drift_report.json())
        
        # Parse drift metrics
        for metric in json_data.get("metrics", []):
            metric_id = metric.get("metric_id") or metric.get("metric", "")
            value = metric.get("value", None)
            
            # Handle different metric value types
            if isinstance(value, dict):
                for sub_name, sub_val in value.items():
                    if isinstance(sub_val, (int, float)):
                        if "drift" in sub_name.lower() and sub_val > DRIFT_THRESHOLD:
                            drift_detected = True
                            drift_details[f"{metric_id}_{sub_name}"] = sub_val
                            log_message(f"Drift detected in {metric_id}_{sub_name}: {sub_val:.4f} > {DRIFT_THRESHOLD}")
            
            elif isinstance(value, (int, float)):
                if "drift" in str(metric_id).lower() and value > DRIFT_THRESHOLD:
                    drift_detected = True
                    drift_details[metric_id] = value
                    log_message(f"Drift detected in {metric_id}: {value:.4f} > {DRIFT_THRESHOLD}")
            
            # Handle column-wise drift
            elif "ValueDrift(column=" in str(metric_id):
                try:
                    col_name = str(metric_id).split("ValueDrift(column=")[1].split(",")[0]
                    if isinstance(value, (int, float)) and value > DRIFT_THRESHOLD:
                        drift_detected = True
                        drift_details[f"{col_name}_drift"] = value
                        log_message(f"Drift detected in column {col_name}: {value:.4f} > {DRIFT_THRESHOLD}")
                except Exception as e:
                    log_message(f"Could not parse drift metric: {metric_id} — {e}")
        
        if not drift_detected:
            log_message(f"No drift detected using {method} method (threshold: {DRIFT_THRESHOLD})")
        
        return drift_detected, drift_details
        
    except Exception as e:
        log_message(f"Error in drift detection with {method}: {e}")
        return False, {}

def save_drift_flag(drift_detected, drift_details, drift_results):
    """Save drift detection result to flag file."""
    try:
        drift_info = {
            "drift_detected": drift_detected,
            "timestamp": datetime.now().isoformat(),
            "drift_threshold": DRIFT_THRESHOLD,
            "methods_used": DRIFT_METHODS,
            "individual_results": drift_results,
            "drift_details": drift_details,
            "flag_message": "DRIFT DETECTED - IMMEDIATE ATTENTION REQUIRED" if drift_detected else "NO DRIFT DETECTED - SYSTEM STABLE"
        }
        
        with open(DRIFT_FLAG_FILE, "w", encoding='utf-8') as f:
            json.dump(drift_info, f, indent=2)
        
        log_message(f"Drift flag saved to: {DRIFT_FLAG_FILE}")
        
        # Also create a simple text flag for easy monitoring
        flag_status = "DRIFT_DETECTED" if drift_detected else "NO_DRIFT"
        with open(DRIFT_STATUS_FLAG, "w") as f:
            f.write(f"{flag_status}\n{datetime.now().isoformat()}")
        
        log_message(f"Status flag saved to: {DRIFT_STATUS_FLAG}")
        
        return drift_info
        
    except Exception as e:
        log_message(f"Error saving drift flag: {e}")
        return None

def main():
    """Main function to run drift detection."""
    log_message("=" * 60)
    log_message("STARTING DRIFT DETECTION PROCESS")
    log_message("=" * 60)
    
    try:
        # Load and validate data
        log_message("Loading datasets...")
        log_message(f"Input directory: {INPUT_DIR}")
        log_message(f"Output files will be saved in: {INPUT_DIR}")
        reference_df, current_df = load_and_validate_data(REFERENCE_CSV, CURRENT_CSV)
        
        # Identify feature types
        numerical_features, categorical_features = identify_feature_types(reference_df)
        
        # Log basic statistics
        log_basic_statistics(reference_df, current_df)
        
        # Track drift detection across multiple methods
        drift_detected_overall = False
        drift_results = {}
        all_drift_details = {}
        
        # Run drift detection for each method
        log_message("\n=== DRIFT DETECTION RESULTS ===")
        for method in DRIFT_METHODS:
            try:
                drift_detected, drift_details = detect_drift_with_evidently(reference_df, current_df, method=method)
                drift_results[method] = drift_detected
                all_drift_details[method] = drift_details
                
                if drift_detected:
                    drift_detected_overall = True
                    log_message(f"⚠️  DRIFT DETECTED using {method.upper()} method!")
                else:
                    log_message(f"✅ No drift detected using {method.upper()} method")
                
            except Exception as e:
                log_message(f"❌ Error with {method} method: {e}")
                drift_results[method] = False
                all_drift_details[method] = {}
        
        # Log final results
        log_message("\n=== FINAL DRIFT DETECTION SUMMARY ===")
        log_message(f"Overall drift status: {'🚨 DRIFT DETECTED' if drift_detected_overall else '✅ NO DRIFT DETECTED'}")
        log_message(f"Methods results: {drift_results}")
        log_message(f"Drift threshold used: {DRIFT_THRESHOLD}")
        
        # Save drift flag
        log_message("Saving drift detection results...")
        drift_info = save_drift_flag(drift_detected_overall, all_drift_details, drift_results)
        
        if drift_detected_overall:
            log_message("🚨 ALERT: Data drift has been detected! Check the drift reports for details.")
            log_message("🔍 Recommended actions:")
            log_message("   - Review model performance")
            log_message("   - Consider model retraining")
            log_message("   - Investigate data source changes")
            
            # Raise an exception/flag for monitoring systems
            raise RuntimeError("DATA DRIFT DETECTED - SYSTEM FLAGGED FOR REVIEW")
        else:
            log_message("✅ System stable - no drift detected")
        
        log_message("Drift detection process completed successfully")
        
    except Exception as e:
        error_msg = f"Error in drift detection process: {e}"
        log_message(f"❌ {error_msg}")
        
        # Save error to drift flag
        try:
            error_info = {
                "drift_detected": None,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "ERROR"
            }
            with open(DRIFT_FLAG_FILE, "w", encoding='utf-8') as f:
                json.dump(error_info, f, indent=2)
        except:
            pass
        
        raise

if __name__ == "__main__":
    main()