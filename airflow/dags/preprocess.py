# Data Preprocessing for Lead Scoring
# Airflow-compatible version (no os, no logging)

# Import only essential libraries
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")

# Configuration - MAKE SURE THESE PATHS ARE ACCESSIBLE
# Configuration - MAKE SURE THESE PATHS ARE ACCESSIBLE
INPUT_PATH = "/opt/airflow/data/Lead Scoring.csv"
OUTPUT_PATH = "/opt/airflow/data"

print(f"Input file: {INPUT_PATH}")
print(f"Output folder: {OUTPUT_PATH}")

# Load Data
def load_data(input_path):
    """Load data from CSV file"""
    print(f"🔄 Loading data from: {input_path}")
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

# Load the dataset
df = load_data(INPUT_PATH)

# Basic Data Exploration
print(f"\n📈 Dataset Overview:")
print(f"  • Shape: {df.shape}")
print(f"  • Columns: {len(df.columns)}")

print(f"\n🔍 First 3 rows:")
print(df.head(3))

print(f"\n🎯 Target Variable:")
if 'Converted' in df.columns:
    print(df['Converted'].value_counts())
    conversion_rate = (df['Converted'].sum() / len(df)) * 100
    print(f"  • Conversion Rate: {conversion_rate:.2f}%")
else:
    print("❌ No 'Converted' column found")

# Missing Values Analysis
print(f"\n🔍 Missing Values Analysis:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing_Count': missing_data,
    'Missing_Percentage': missing_percent
}).sort_values('Missing_Count', ascending=False)

# Show only columns with missing values
missing_cols = missing_summary[missing_summary['Missing_Count'] > 0]
if len(missing_cols) > 0:
    print(missing_cols)
else:
    print("✅ No missing values found!")

# Drop Unnecessary Columns
def drop_unnecessary_columns(df):
    """Drop columns that are not useful for modeling"""
    drop_cols = [
        'Prospect ID', 'Lead Number', 'Magazine', 'Receive More Updates About Our Courses',
        'Update me on Supply Chain Content', 'Get updates on DM Content',
        'I agree to pay the amount through cheque', 'Search', 'Newspaper Article',
        'X Education Forums', 'Newspaper', 'Digital Advertisement', 'Through Recommendations'
    ]
    
    # Only drop columns that actually exist
    existing_drop_cols = [col for col in drop_cols if col in df.columns]
    df_cleaned = df.drop(columns=existing_drop_cols)
    
    print(f"🗑️ Dropped {len(existing_drop_cols)} unnecessary columns")
    if existing_drop_cols:
        print(f"   Dropped: {existing_drop_cols}")
    
    return df_cleaned

# Apply column dropping
df_cleaned = drop_unnecessary_columns(df)
print(f"✅ Remaining columns: {len(df_cleaned.columns)}")

# Handle Missing Values
def fill_missing_values(df):
    """Fill missing values with appropriate strategies"""
    df_filled = df.copy()
    
    # Numerical columns - fill with median
    numerical_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit',
                      'Asymmetrique Activity Score', 'Asymmetrique Profile Score']
    
    print(f"\n🔢 Handling Numerical Columns:")
    for col_name in numerical_cols:
        if col_name in df_filled.columns:
            missing_count = df_filled[col_name].isnull().sum()
            if missing_count > 0:
                median_val = df_filled[col_name].median()
                df_filled[col_name].fillna(median_val, inplace=True)
                print(f"  • {col_name}: {missing_count} values → median ({median_val:.2f})")
    
    # Binary columns - fill with 'No'
    binary_cols = ['Do Not Email', 'Do Not Call', 'A free copy of Mastering The Interview']
    print(f"\n🔘 Handling Binary Columns:")
    for col_name in binary_cols:
        if col_name in df_filled.columns:
            missing_count = df_filled[col_name].isnull().sum()
            if missing_count > 0:
                df_filled[col_name].fillna('No', inplace=True)
                print(f"  • {col_name}: {missing_count} values → 'No'")
    
    # Categorical columns - fill with mode or 'Unknown'
    categorical_cols = [
        'Tags', 'Lead Quality', 'Lead Profile', 'What is your current occupation',
        'Last Activity', 'Last Notable Activity', 'Lead Source', 'Lead Origin',
        'Specialization', 'City', 'How did you hear about X Education',
        'What matters most to you in choosing a course', 'Country'
    ]
    
    print(f"\n🏷️ Handling Categorical Columns:")
    for col_name in categorical_cols:
        if col_name in df_filled.columns:
            missing_count = df_filled[col_name].isnull().sum()
            if missing_count > 0:
                # Try to use mode, fallback to 'Unknown'
                mode_val = df_filled[col_name].mode()
                fill_value = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                df_filled[col_name].fillna(fill_value, inplace=True)
                print(f"  • {col_name}: {missing_count} values → '{fill_value}'")
    
    return df_filled

# Apply missing value handling
df_filled = fill_missing_values(df_cleaned)
remaining_missing = df_filled.isnull().sum().sum()
print(f"\n✅ Missing value handling complete. Remaining missing: {remaining_missing}")

# Create Feature Engineering Pipeline
def create_preprocessing_pipeline(df):
    """Create sklearn preprocessing pipeline"""
    
    # Define column categories
    numerical_cols = []
    categorical_cols = []
    
    for col in df.columns:
        if col == 'Converted':  # Skip target column
            continue
        elif df[col].dtype in ['int64', 'float64']:
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    print(f"\n🔧 Pipeline Configuration:")
    print(f"  • Numerical columns ({len(numerical_cols)}): {numerical_cols}")
    print(f"  • Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Create transformers
    transformers = []
    
    if numerical_cols:
        transformers.append(('num', MinMaxScaler(), numerical_cols))
        print(f"  ✅ Added MinMax scaler for numerical features")
    
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
        print(f"  ✅ Added OneHot encoder for categorical features")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any remaining columns
    )
    
    return preprocessor, numerical_cols, categorical_cols

# Create pipeline
preprocessor, numerical_cols, categorical_cols = create_preprocessing_pipeline(df_filled)

# Apply Preprocessing
def apply_preprocessing(df, preprocessor):
    """Apply the preprocessing pipeline"""
    print(f"\n🔄 Applying preprocessing pipeline...")
    
    # Separate features and target
    if 'Converted' in df.columns:
        X = df.drop('Converted', axis=1)
        y = df['Converted']
        has_target = True
    else:
        X = df
        y = None
        has_target = False
    
    # Transform features
    X_transformed = preprocessor.fit_transform(X)
    print(f"✅ Features transformed. Shape: {X_transformed.shape}")
    
    # Get feature names after transformation
    feature_names = []
    
    # Add numerical feature names
    if numerical_cols:
        feature_names.extend(numerical_cols)
    
    # Add categorical feature names
    if categorical_cols:
        try:
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_features)
        except Exception as e:
            print(f"⚠️ Could not get categorical feature names: {e}")
            # Fallback: create generic names
            remaining_features = X_transformed.shape[1] - len(numerical_cols)
            feature_names.extend([f"cat_feature_{i}" for i in range(remaining_features)])
    
    # Create final DataFrame
    df_final = pd.DataFrame(X_transformed, columns=feature_names, index=df.index)
    
    # Add target back if it exists
    if has_target:
        df_final['Converted'] = y
        print(f"✅ Target column added back")
    
    print(f"✅ Final dataset shape: {df_final.shape}")
    return df_final

# Apply preprocessing
df_processed = apply_preprocessing(df_filled, preprocessor)

# Data Summary
print(f"\n📊 FINAL DATASET SUMMARY:")
print(f"  • Total samples: {len(df_processed)}")
print(f"  • Total features: {len(df_processed.columns) - (1 if 'Converted' in df_processed.columns else 0)}")
print(f"  • Has target: {'Yes' if 'Converted' in df_processed.columns else 'No'}")

if 'Converted' in df_processed.columns:
    target_dist = df_processed['Converted'].value_counts()
    print(f"\n🎯 Target Distribution:")
    print(f"  • Class 0: {target_dist[0]} ({target_dist[0]/len(df_processed)*100:.1f}%)")
    print(f"  • Class 1: {target_dist[1]} ({target_dist[1]/len(df_processed)*100:.1f}%)")

# Save Results
def save_processed_data(df_processed, preprocessor, output_path):
    """Save processed data and pipeline"""
    print(f"\n💾 Saving results to: {output_path}")
    
    try:
        # Save processed data
        data_file = f"{output_path}/preprocessed_data.csv"
        df_processed.to_csv(data_file, index=False)
        print(f"✅ Processed data saved: {data_file}")
        
        # Save preprocessing pipeline
        pipeline_file = f"{output_path}/preprocessing_pipeline.pkl"
        with open(pipeline_file, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"✅ Pipeline saved: {pipeline_file}")
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': df.shape,
            'processed_shape': df_processed.shape,
            'columns': list(df_processed.columns),
            'has_target': 'Converted' in df_processed.columns,
            'feature_count': len(df_processed.columns) - (1 if 'Converted' in df_processed.columns else 0)
        }
        
        metadata_file = f"{output_path}/processing_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Metadata saved: {metadata_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving files: {str(e)}")
        print(f"   This might be due to Docker path access limitations")
        return False

# Save everything
save_success = save_processed_data(df_processed, preprocessor, OUTPUT_PATH)

# Final Status
print(f"\n🎉 PREPROCESSING COMPLETED!")
print(f"📁 Output location: {OUTPUT_PATH}")

if save_success:
    print(f"✅ All files saved successfully")
else:
    print(f"⚠️ Files could not be saved (but processing completed)")

print(f"\n🤖 MODEL READINESS STATUS:")
print(f"✅ Data cleaned and preprocessed")
print(f"✅ Missing values handled")
print(f"✅ Features scaled and encoded")
print(f"✅ Ready for machine learning!")

# Show final sample
print(f"\n📋 Final Data Sample (first 3 rows):")
print(df_processed.head(3))

print(f"\n📈 Basic Statistics:")
print(df_processed.describe())