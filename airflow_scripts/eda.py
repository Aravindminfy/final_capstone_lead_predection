# Comprehensive EDA for Sales Conversion Prediction
# Production-Level Analysis with Business and Technical Insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import missingno as msno

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

class SalesConversionEDA:
    """
    Comprehensive EDA class for Sales Conversion Prediction
    Includes both business and technical analysis
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.target = 'Converted'
        self.numerical_features = []
        self.categorical_features = []
        self.business_insights = {}
        self.technical_insights = {}
        
    def initial_data_overview(self):
        """
        Initial data exploration and overview
        """
        print("="*80)
        print("INITIAL DATA OVERVIEW")
        print("="*80)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Total Records: {self.df.shape[0]:,}")
        print(f"Total Features: {self.df.shape[1]:,}")
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        # Target variable distribution
        print(f"\nTarget Variable Distribution:")
        target_dist = self.df[self.target].value_counts()
        print(target_dist)
        print(f"Conversion Rate: {target_dist[1]/target_dist.sum()*100:.2f}%")
        
        # Store business insight
        self.business_insights['conversion_rate'] = target_dist[1]/target_dist.sum()*100
        
    def data_quality_assessment(self):
        """
        Comprehensive data quality assessment
        """
        print("="*80)
        print("DATA QUALITY ASSESSMENT")
        print("="*80)
        
        # Missing values analysis
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        print("Missing Values Summary:")
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Visualize missing data
        plt.figure(figsize=(15, 8))
        msno.matrix(self.df)
        plt.title('Missing Data Pattern')
        plt.tight_layout()
        plt.show()
        
        # Duplicate records
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate Records: {duplicates}")
        
        # Data types analysis
        print("\nData Types Distribution:")
        dtype_counts = self.df.dtypes.value_counts()
        print(dtype_counts)
        
        # Store technical insights
        self.technical_insights['missing_data'] = missing_df
        self.technical_insights['duplicates'] = duplicates
        
    def feature_categorization(self):
        """
        Categorize features into numerical and categorical
        """
        print("="*80)
        print("FEATURE CATEGORIZATION")
        print("="*80)
        
        # Identify numerical and categorical features
        self.numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target and ID columns
        if self.target in self.numerical_features:
            self.numerical_features.remove(self.target)
        if 'Prospect ID' in self.categorical_features:
            self.categorical_features.remove('Prospect ID')
        if 'Lead Number' in self.numerical_features:
            self.numerical_features.remove('Lead Number')
            
        print(f"Numerical Features ({len(self.numerical_features)}):")
        print(self.numerical_features)
        
        print(f"\nCategorical Features ({len(self.categorical_features)}):")
        print(self.categorical_features)
        
    def univariate_analysis(self):
        """
        Detailed univariate analysis for all features
        """
        print("="*80)
        print("UNIVARIATE ANALYSIS")
        print("="*80)
        
        # Target variable analysis
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        target_counts = self.df[self.target].value_counts()
        plt.pie(target_counts.values, labels=['Not Converted', 'Converted'], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Target Variable Distribution')
        
        plt.subplot(1, 2, 2)
        sns.countplot(data=self.df, x=self.target)
        plt.title('Target Variable Count')
        plt.tight_layout()
        plt.show()
        
        # Numerical features analysis
        if self.numerical_features:
            n_cols = 3
            n_rows = (len(self.numerical_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for i, feature in enumerate(self.numerical_features):
                if i < len(axes):
                    # Distribution plot
                    sns.histplot(self.df[feature].dropna(), kde=True, ax=axes[i])
                    axes[i].set_title(f'{feature} Distribution')
                    axes[i].tick_params(axis='x', rotation=45)
                    
                    # Add statistics
                    mean_val = self.df[feature].mean()
                    median_val = self.df[feature].median()
                    axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                    axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
                    axes[i].legend()
            
            # Hide empty subplots
            for i in range(len(self.numerical_features), len(axes)):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            plt.show()
        
        # Categorical features analysis (top categories)
        if self.categorical_features:
            for feature in self.categorical_features[:6]:  # Show first 6 categorical features
                plt.figure(figsize=(12, 6))
                
                # Count plot
                top_categories = self.df[feature].value_counts().head(10)
                
                plt.subplot(1, 2, 1)
                sns.countplot(data=self.df, y=feature, order=top_categories.index)
                plt.title(f'{feature} - Top 10 Categories')
                
                # Pie chart for top categories
                plt.subplot(1, 2, 2)
                if len(top_categories) <= 8:
                    plt.pie(top_categories.values, labels=top_categories.index, autopct='%1.1f%%')
                else:
                    other_sum = self.df[feature].value_counts().iloc[8:].sum()
                    pie_data = top_categories.iloc[:8].tolist() + [other_sum]
                    pie_labels = top_categories.iloc[:8].index.tolist() + ['Others']
                    plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%')
                
                plt.title(f'{feature} Distribution')
                plt.tight_layout()
                plt.show()
                
    def bivariate_analysis(self):
        """
        Comprehensive bivariate analysis with target variable
        """
        print("="*80)
        print("BIVARIATE ANALYSIS")
        print("="*80)
        
        # Numerical features vs Target
        if self.numerical_features:
            n_cols = 2
            n_rows = (len(self.numerical_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for i, feature in enumerate(self.numerical_features):
                if i < len(axes):
                    # Box plot
                    sns.boxplot(data=self.df, x=self.target, y=feature, ax=axes[i])
                    axes[i].set_title(f'{feature} vs Conversion')
                    
                    # Statistical test
                    converted = self.df[self.df[self.target] == 1][feature].dropna()
                    not_converted = self.df[self.df[self.target] == 0][feature].dropna()
                    
                    if len(converted) > 0 and len(not_converted) > 0:
                        stat, p_value = stats.mannwhitneyu(converted, not_converted, alternative='two-sided')
                        axes[i].text(0.5, 0.95, f'p-value: {p_value:.4f}', 
                                   transform=axes[i].transAxes, ha='center', va='top')
            
            # Hide empty subplots
            for i in range(len(self.numerical_features), len(axes)):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            plt.show()
        
        # Categorical features vs Target
        significant_associations = []
        
        for feature in self.categorical_features[:8]:  # Analyze first 8 categorical features
            plt.figure(figsize=(15, 6))
            
            # Cross-tabulation
            crosstab = pd.crosstab(self.df[feature], self.df[self.target])
            
            # Conversion rate by category
            conversion_rate = crosstab.div(crosstab.sum(axis=1), axis=0)[1] * 100
            
            plt.subplot(1, 3, 1)
            sns.countplot(data=self.df, x=self.target, hue=feature)
            plt.title(f'{feature} vs Conversion - Count')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.subplot(1, 3, 2)
            crosstab.plot(kind='bar', ax=plt.gca())
            plt.title(f'{feature} vs Conversion - Stacked')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 3, 3)
            conversion_rate.plot(kind='bar', color='skyblue')
            plt.title(f'Conversion Rate by {feature}')
            plt.ylabel('Conversion Rate (%)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            # Chi-square test
            try:
                chi2, p_value, dof, expected = chi2_contingency(crosstab)
                if p_value < 0.05:
                    significant_associations.append((feature, p_value, chi2))
                    print(f"{feature}: Chi-square = {chi2:.4f}, p-value = {p_value:.4f} (Significant)")
                else:
                    print(f"{feature}: Chi-square = {chi2:.4f}, p-value = {p_value:.4f} (Not Significant)")
            except:
                print(f"{feature}: Chi-square test could not be performed")
        
        self.business_insights['significant_associations'] = significant_associations
        
    def correlation_analysis(self):
        """
        Correlation analysis for numerical features
        """
        print("="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        if len(self.numerical_features) > 1:
            # Correlation matrix
            numerical_df = self.df[self.numerical_features + [self.target]]
            correlation_matrix = numerical_df.corr()
            
            # Heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()
            
            # Correlation with target
            target_corr = correlation_matrix[self.target].drop(self.target).sort_values(key=abs, ascending=False)
            print("\nCorrelation with Target Variable:")
            print(target_corr)
            
            # High correlation pairs (multicollinearity check)
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.7:
                        high_corr_pairs.append((
                            correlation_matrix.columns[i], 
                            correlation_matrix.columns[j], 
                            correlation_matrix.iloc[i, j]
                        ))
            
            if high_corr_pairs:
                print("\nHigh Correlation Pairs (>0.7):")
                for pair in high_corr_pairs:
                    print(f"{pair[0]} - {pair[1]}: {pair[2]:.4f}")
            
            self.technical_insights['high_correlation'] = high_corr_pairs
            
    def advanced_analysis(self):
        """
        Advanced analysis including feature importance and business insights
        """
        print("="*80)
        print("ADVANCED ANALYSIS")
        print("="*80)
        
        # Feature importance using mutual information
        # Prepare data for mutual information
        df_encoded = self.df.copy()
        
        # Label encode categorical variables
        le = LabelEncoder()
        for feature in self.categorical_features:
            df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
        
        # Calculate mutual information
        features = self.numerical_features + self.categorical_features
        X = df_encoded[features].fillna(0)
        y = df_encoded[self.target]
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_scores = pd.Series(mi_scores, index=features).sort_values(ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        mi_scores.head(15).plot(kind='barh')
        plt.title('Feature Importance (Mutual Information)')
        plt.xlabel('Mutual Information Score')
        plt.tight_layout()
        plt.show()
        
        print("Top 10 Most Important Features:")
        print(mi_scores.head(10))
        
        # Business insights analysis
        self.generate_business_insights()
        
        self.technical_insights['feature_importance'] = mi_scores
        
    def generate_business_insights(self):
        """
        Generate business-specific insights
        """
        print("="*80)
        print("BUSINESS INSIGHTS")
        print("="*80)
        
        # Lead source analysis
        if 'Lead Source' in self.df.columns:
            lead_source_conversion = self.df.groupby('Lead Source')[self.target].agg(['count', 'sum', 'mean'])
            lead_source_conversion.columns = ['Total_Leads', 'Conversions', 'Conversion_Rate']
            lead_source_conversion = lead_source_conversion.sort_values('Conversion_Rate', ascending=False)
            
            print("Lead Source Performance:")
            print(lead_source_conversion)
            
            # Visualize lead source performance
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            lead_source_conversion['Total_Leads'].plot(kind='bar')
            plt.title('Total Leads by Source')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            lead_source_conversion['Conversion_Rate'].plot(kind='bar', color='green')
            plt.title('Conversion Rate by Lead Source')
            plt.xticks(rotation=45)
            plt.ylabel('Conversion Rate')
            
            plt.tight_layout()
            plt.show()
            
            self.business_insights['lead_source_performance'] = lead_source_conversion
        
        # Website engagement analysis
        website_features = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
        website_features = [f for f in website_features if f in self.df.columns]
        
        if website_features:
            print("\nWebsite Engagement Analysis:")
            for feature in website_features:
                converted_avg = self.df[self.df[self.target] == 1][feature].mean()
                not_converted_avg = self.df[self.df[self.target] == 0][feature].mean()
                
                print(f"{feature}:")
                print(f"  Converted: {converted_avg:.2f}")
                print(f"  Not Converted: {not_converted_avg:.2f}")
                print(f"  Difference: {((converted_avg - not_converted_avg) / not_converted_avg * 100):.2f}%")
        
        # Activity-based insights
        if 'Last Activity' in self.df.columns:
            activity_conversion = self.df.groupby('Last Activity')[self.target].agg(['count', 'mean'])
            activity_conversion.columns = ['Count', 'Conversion_Rate']
            activity_conversion = activity_conversion.sort_values('Conversion_Rate', ascending=False)
            
            print("\nLast Activity Performance:")
            print(activity_conversion.head(10))
            
            self.business_insights['activity_performance'] = activity_conversion
            
    def outlier_analysis(self):
        """
        Comprehensive outlier analysis
        """
        print("="*80)
        print("OUTLIER ANALYSIS")
        print("="*80)
        
        if self.numerical_features:
            outlier_summary = {}
            
            for feature in self.numerical_features:
                Q1 = self.df[feature].quantile(0.25)
                Q3 = self.df[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
                outlier_percentage = (len(outliers) / len(self.df)) * 100
                
                outlier_summary[feature] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                print(f"{feature}: {len(outliers)} outliers ({outlier_percentage:.2f}%)")
            
            # Visualize outliers
            n_cols = 2
            n_rows = (len(self.numerical_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for i, feature in enumerate(self.numerical_features):
                if i < len(axes):
                    sns.boxplot(data=self.df, y=feature, ax=axes[i])
                    axes[i].set_title(f'{feature} - Outlier Detection')
            
            # Hide empty subplots
            for i in range(len(self.numerical_features), len(axes)):
                axes[i].set_visible(False)
                
            plt.tight_layout()
            plt.show()
            
            self.technical_insights['outliers'] = outlier_summary
            
    def generate_final_report(self):
        """
        Generate comprehensive final report
        """
        print("="*100)
        print("COMPREHENSIVE EDA REPORT - SALES CONVERSION PREDICTION")
        print("="*100)
        
        print("\n1. DATASET OVERVIEW:")
        print(f"   - Total Records: {self.df.shape[0]:,}")
        print(f"   - Total Features: {self.df.shape[1]:,}")
        print(f"   - Conversion Rate: {self.business_insights.get('conversion_rate', 0):.2f}%")
        
        print("\n2. DATA QUALITY:")
        missing_df = self.technical_insights.get('missing_data', pd.DataFrame())
        if not missing_df.empty:
            high_missing = missing_df[missing_df['Missing_Percentage'] > 30]
            print(f"   - Features with >30% missing values: {len(high_missing)}")
            
        print(f"   - Duplicate records: {self.technical_insights.get('duplicates', 0)}")
        
        print("\n3. KEY BUSINESS INSIGHTS:")
        
        # Lead source insights
        if 'lead_source_performance' in self.business_insights:
            best_source = self.business_insights['lead_source_performance'].iloc[0]
            print(f"   - Best performing lead source: {best_source.name} ({best_source['Conversion_Rate']:.2f}% conversion)")
        
        # Feature importance
        if 'feature_importance' in self.technical_insights:
            top_features = self.technical_insights['feature_importance'].head(5)
            print(f"   - Top 5 predictive features: {', '.join(top_features.index)}")
        
        # Significant associations
        if 'significant_associations' in self.business_insights:
            sig_count = len(self.business_insights['significant_associations'])
            print(f"   - Features with significant association with conversion: {sig_count}")
        
        print("\n4. TECHNICAL RECOMMENDATIONS:")
        
        # Missing value treatment
        if not missing_df.empty:
            high_missing = missing_df[missing_df['Missing_Percentage'] > 50]
            if not high_missing.empty:
                print(f"   - Consider removing features with >50% missing values: {list(high_missing['Column'])}")
        
        # Multicollinearity
        if 'high_correlation' in self.technical_insights:
            high_corr = self.technical_insights['high_correlation']
            if high_corr:
                print(f"   - Address multicollinearity between: {high_corr[0][0]} and {high_corr[0][1]}")
        
        # Outliers
        if 'outliers' in self.technical_insights:
            outlier_features = [f for f, info in self.technical_insights['outliers'].items() 
                              if info['outlier_percentage'] > 5]
            if outlier_features:
                print(f"   - Consider outlier treatment for: {', '.join(outlier_features)}")
        
        print("\n5. MODELING RECOMMENDATIONS:")
        print("   - Use stratified sampling due to class imbalance")
        print("   - Consider ensemble methods (Random Forest, XGBoost)")
        print("   - Implement proper cross-validation")
        print("   - Focus on precision and recall along with accuracy")
        
        print("\n" + "="*100)

