# app.py - Flask Lead Scoring Prediction App with Fixed Pipeline Logic

from flask import Flask, request, render_template, send_file, flash, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import warnings
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_this'

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# File paths - Updated to match your working paths
PIPELINE_PATH = r"C:\Users\Minfy.DESKTOP-3E50D5N\Desktop\final_capstone\preprocess\preprocessed_output\processed\pipeline_model.pkl"
METADATA_PATH = r"C:\Users\Minfy.DESKTOP-3E50D5N\Desktop\final_capstone\preprocess\preprocessed_output\processed\metadata.pkl"
MODEL_PATH = r"C:\Users\Minfy.DESKTOP-3E50D5N\Desktop\final_capstone\modal_development\model\best_model_xgboost.pkl"

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

class LeadConversionPipeline:
    """
    A pipeline for lead conversion prediction - Using exact working logic
    """

    def __init__(self, pipeline_path: str, metadata_path: str, model_path: str):
        """
        Initialize the pipeline by loading all required components.
        """
        self.pipeline_path = pipeline_path
        self.metadata_path = metadata_path
        self.model_path = model_path

        self.preprocessor = None
        self.metadata = None
        self.model = None
        self.model_data = None

        self._load_components()

    def _load_components(self) -> None:
        """Loads the preprocessing pipeline, metadata, and trained model from disk."""
        try:
            with open(self.pipeline_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            logger.info("Preprocessing pipeline loaded successfully")

            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info("Metadata loaded successfully")

            with open(self.model_path, 'rb') as f:
                self.model_data = pickle.load(f)

            if isinstance(self.model_data, dict):
                for key in ['best_estimator_', 'model', 'estimator', 'best_model', 'classifier']:
                    if key in self.model_data:
                        self.model = self.model_data[key]
                        logger.info(f"Model loaded from key: '{key}'")
                        break
                if self.model is None:
                    for key, value in self.model_data.items():
                        if hasattr(value, 'predict'):
                            self.model = value
                            break
            else:
                self.model = self.model_data

            if not hasattr(self.model, 'predict'):
                raise ValueError("Loaded object is not a valid model")

            logger.info(f"Model type: {type(self.model)}")

        except Exception as e:
            logger.error(f"Error loading components: {e}")
            raise

    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the input DataFrame to correct data types and missing values."""
        df_clean = df.copy()
        numerical_cols = self.metadata['numerical_features']
        categorical_cols = self.metadata['categorical_features']

        for col in numerical_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col].astype(str), errors='coerce')
                if df_clean[col].isna().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)

        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).replace(['nan', 'None', 'null'], 'Unknown').str.strip()

        return df_clean

    def _encode_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes binary categorical values into numerical format."""
        df_encoded = df.copy()
        binary_cols = self.metadata['binary_features']

        for col in binary_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].astype(str).str.strip().str.lower()
                binary_mapping = {
                    'yes': 1, 'y': 1, '1': 1, 'true': 1, 'on': 1,
                    'no': 0, 'n': 0, '0': 0, 'false': 0, 'off': 0,
                    'unknown': 0, 'nan': 0, 'none': 0
                }
                df_encoded[f'{col}_encoded'] = df_encoded[col].map(binary_mapping).fillna(0).astype(int)

        return df_encoded

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aligns feature columns to match those used in model training."""
        if 'feature_names' in self.model_data:
            expected_features = self.model_data['feature_names']
        elif hasattr(self.model, 'feature_names_in_'):
            expected_features = self.model.feature_names_in_
        else:
            return X

        X_aligned = pd.DataFrame(0, index=X.index, columns=expected_features)

        for col in X.columns:
            if col in expected_features:
                X_aligned[col] = X[col]

        return X_aligned

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Runs the entire preprocessing pipeline on the input data."""
        df_processed = df.copy()
        actual_labels = None

        if 'Converted' in df_processed.columns:
            actual_labels = df_processed['Converted']
            df_processed.drop(columns=['Converted'], inplace=True)

        df_processed = self._clean_data_types(df_processed)
        df_processed = self._encode_binary_features(df_processed)

        numerical_cols = self.metadata['numerical_features']
        categorical_cols = self.metadata['categorical_features']
        binary_cols = self.metadata['binary_features']

        available_num_cols = [col for col in numerical_cols if col in df_processed.columns]
        available_cat_cols = [col for col in categorical_cols if col in df_processed.columns]

        if available_num_cols or available_cat_cols:
            X_sample = df_processed[available_num_cols + available_cat_cols]
            X_transformed = self.preprocessor.transform(X_sample)

            try:
                feature_names = self.preprocessor.get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

            X_final = pd.DataFrame(X_transformed, columns=feature_names, index=df_processed.index)

            for col in [f"{col}_encoded" for col in binary_cols if col in df.columns]:
                if col in df_processed.columns:
                    X_final[col] = df_processed[col].values

            X_final = self._align_features(X_final)
            return X_final, actual_labels

        else:
            raise ValueError("No valid numerical or categorical columns found for preprocessing")

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates predictions for the given input data."""
        X_final, actual_labels = self.preprocess_data(df)
        predictions = self.model.predict(X_final)

        try:
            probabilities = self.model.predict_proba(X_final)
        except:
            probabilities = None

        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'actual_labels': actual_labels,
            'input_data': df.copy()
        }

        if actual_labels is not None:
            accuracy = (predictions == actual_labels).mean() * 100
            results['accuracy'] = accuracy

        return results

    def predict_from_file(self, file_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Reads data from a CSV file, makes predictions, and optionally saves results to CSV."""
        df = pd.read_csv(file_path)
        results = self.predict(df)

        results_df = results['input_data'].copy()
        results_df['Predicted_Converted'] = results['predictions']

        if results['probabilities'] is not None:
            prob_df = pd.DataFrame(results['probabilities'],
                                   columns=[f'Prob_Class_{i}' for i in range(results['probabilities'].shape[1])],
                                   index=results_df.index)
            results_df = pd.concat([results_df, prob_df], axis=1)

        if results['actual_labels'] is not None:
            results_df['Actual_Converted'] = results['actual_labels']
            results_df['Prediction_Correct'] = (results_df['Predicted_Converted'] == results_df['Actual_Converted'])

        if output_path:
            results_df.to_csv(output_path, index=False)

        results['results_df'] = results_df
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata and details about the loaded model."""
        info = {
            'model_type': type(self.model).__name__,
            'numerical_features': self.metadata['numerical_features'],
            'categorical_features': self.metadata['categorical_features'],
            'binary_features': self.metadata['binary_features']
        }

        if 'feature_names' in self.model_data:
            info['total_features'] = len(self.model_data['feature_names'])
            info['feature_names'] = self.model_data['feature_names']

        return info

# Initialize global pipeline
pipeline = None

def load_pipeline():
    """Load the prediction pipeline"""
    global pipeline
    try:
        pipeline = LeadConversionPipeline(PIPELINE_PATH, METADATA_PATH, MODEL_PATH)
        logger.info("🚀 Pipeline loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load pipeline: {e}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global pipeline
    
    if pipeline is None:
        flash('Pipeline not loaded. Please restart the application.', 'error')
        return redirect(url_for('index'))
    
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload CSV or Excel files.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Read file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        logger.info(f"📊 Loaded data with shape: {df.shape}")
        
        # Make predictions using the working pipeline logic
        results = pipeline.predict(df)
        
        # Prepare results DataFrame with enhanced columns
        results_df = results['input_data'].copy()
        results_df['Lead_Score_Prediction'] = results['predictions']
        results_df['Conversion_Prediction'] = ['High Potential' if pred == 1 else 'Low Potential' 
                                              for pred in results['predictions']]
        
        # Add probabilities if available
        if results['probabilities'] is not None:
            # Add probability columns
            prob_df = pd.DataFrame(results['probabilities'],
                                 columns=[f'Prob_Class_{i}' for i in range(results['probabilities'].shape[1])],
                                 index=results_df.index)
            results_df = pd.concat([results_df, prob_df], axis=1)
            
            # Add conversion probability percentage
            prob_positive = results['probabilities'][:, 1]  # Probability for class 1
            results_df['Conversion_Probability'] = np.round(prob_positive * 100, 2)
            results_df['Lead_Quality_Score'] = pd.cut(prob_positive, 
                                                     bins=[0, 0.3, 0.6, 1.0], 
                                                     labels=['Low', 'Medium', 'High'])
        
        # Save results
        result_filename = f"predictions_{filename}"
        result_filepath = os.path.join(RESULTS_FOLDER, result_filename)
        results_df.to_csv(result_filepath, index=False)
        
        # Generate summary
        total_records = len(results_df)
        high_potential = sum(results['predictions'])
        low_potential = total_records - high_potential
        conversion_rate = (high_potential / total_records) * 100
        
        summary = {
            'total_records': total_records,
            'high_potential': int(high_potential),
            'low_potential': int(low_potential),
            'conversion_rate': round(conversion_rate, 2),
            'result_file': result_filename
        }
        
        # Show sample predictions (last 5 rows)
        sample_predictions = results_df.tail().to_html(classes='table table-striped', 
                                                      table_id='predictions-table')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        flash('Predictions generated successfully!', 'success')
        return render_template('results.html', 
                             summary=summary, 
                             sample_predictions=sample_predictions,
                             filename=result_filename)
        
    except Exception as e:
        logger.error(f"❌ Error processing file: {e}")
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(RESULTS_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            flash('File not found', 'error')
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check endpoint"""
    global pipeline
    return {
        'status': 'healthy' if pipeline is not None else 'unhealthy',
        'pipeline_loaded': pipeline is not None
    }

if __name__ == '__main__':
    print("🚀 Starting Flask Lead Scoring Prediction App...")
    if load_pipeline():
        print("✅ Pipeline loaded successfully. App is ready!")
        print("🌐 Flask App: http://localhost:8090")
        print("🌊 Airflow UI: http://localhost:8080")
        print("📊 MLflow UI: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=8090)
    else:
        print("❌ Failed to load pipeline. Please check file paths and try again.")
        print("Update the file paths in app.py:")
        print(f"  PIPELINE_PATH: {PIPELINE_PATH}")
        print(f"  METADATA_PATH: {METADATA_PATH}")
        print(f"  MODEL_PATH: {MODEL_PATH}")