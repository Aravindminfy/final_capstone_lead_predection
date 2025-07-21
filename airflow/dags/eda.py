import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import io
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sales Conversion Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #ff7f0e;
        padding-left: 1rem;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .recommendation-box {
        background-color: #fff2e6;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ff7f0e;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_uploaded_data(uploaded_file):
    """Load and cache the uploaded dataset"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

class SalesConversionDashboard:
    def __init__(self):
        self.df = None
        self.target = 'Converted'
        self.numerical_features = []
        self.categorical_features = []
        
    def load_data(self):
        """Load and validate the dataset"""
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload your sales conversion dataset with a 'Converted' column"
        )
        
        if uploaded_file is not None:
            self.df = load_uploaded_data(uploaded_file)
            if self.df is not None:
                # Validate required columns
                if self.target not in self.df.columns:
                    st.error(f"❌ Required column '{self.target}' not found in the dataset!")
                    st.info("Please ensure your dataset has a 'Converted' column with 0/1 values")
                    return False
                
                st.success(f"✅ Dataset loaded successfully! Shape: {self.df.shape}")
                
                # Show column info
                with st.expander("📋 Dataset Columns"):
                    col_info = pd.DataFrame({
                        'Column': self.df.columns,
                        'Data Type': self.df.dtypes,
                        'Non-Null Count': self.df.count(),
                        'Null Count': self.df.isnull().sum()
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                return True
            return False
        else:
            st.info("📁 Please upload your sales conversion CSV file to begin analysis")
            return False
    
    def categorize_features(self):
        """Categorize features into numerical and categorical"""
        if self.df is not None:
            self.numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
            
            # Remove target and ID columns
            if self.target in self.numerical_features:
                self.numerical_features.remove(self.target)
            
            # Remove common ID columns
            id_columns = ['Prospect ID', 'Lead Number', 'ID', 'id', 'prospect_id', 'lead_number']
            for col in id_columns:
                if col in self.categorical_features:
                    self.categorical_features.remove(col)
                if col in self.numerical_features:
                    self.numerical_features.remove(col)
    
    def show_overview(self):
        """Display dataset overview"""
        st.markdown('<h2 class="section-header">📈 Dataset Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Records",
                value=f"{self.df.shape[0]:,}",
                help="Total number of records in the dataset"
            )
        
        with col2:
            st.metric(
                label="Total Features",
                value=f"{self.df.shape[1]:,}",
                help="Total number of features/columns"
            )
        
        with col3:
            conversion_rate = (self.df[self.target].sum() / len(self.df)) * 100
            st.metric(
                label="Conversion Rate",
                value=f"{conversion_rate:.2f}%",
                help="Percentage of leads that converted"
            )
        
        with col4:
            st.metric(
                label="Conversions",
                value=f"{self.df[self.target].sum():,}",
                help="Total number of successful conversions"
            )
        
        # Target distribution visualization
        st.subheader("🎯 Target Variable Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            target_counts = self.df[self.target].value_counts()
            fig = px.pie(
                values=target_counts.values,
                names=['Not Converted', 'Converted'],
                title="Conversion Distribution",
                color_discrete_sequence=['#ff9999', '#66b3ff']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=['Not Converted', 'Converted'],
                y=target_counts.values,
                title="Conversion Counts",
                color=['Not Converted', 'Converted'],
                color_discrete_sequence=['#ff9999', '#66b3ff']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sample data
        st.subheader("📊 Sample Data")
        st.dataframe(self.df.head(10), use_container_width=True)
        
        # Feature categorization
        with st.expander("🔍 Feature Categories"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Numerical Features ({len(self.numerical_features)})")
                if self.numerical_features:
                    for feature in self.numerical_features:
                        st.write(f"• {feature}")
                else:
                    st.write("No numerical features found")
            
            with col2:
                st.subheader(f"Categorical Features ({len(self.categorical_features)})")
                if self.categorical_features:
                    for feature in self.categorical_features:
                        st.write(f"• {feature}")
                else:
                    st.write("No categorical features found")
    
    def show_data_quality(self):
        """Display data quality assessment"""
        st.markdown('<h2 class="section-header">🔍 Data Quality Assessment</h2>', unsafe_allow_html=True)
        
        # Missing values analysis
        st.subheader("Missing Values Analysis")
        
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Feature': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percent.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        missing_features = missing_df[missing_df['Missing_Count'] > 0]
        
        if len(missing_features) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(missing_features, use_container_width=True)
            
            with col2:
                if len(missing_features) > 0:
                    fig = px.bar(
                        missing_features.head(10),
                        x='Missing_Percentage',
                        y='Feature',
                        orientation='h',
                        title="Top Features with Missing Values",
                        labels={'Missing_Percentage': 'Missing %'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
        
        # Data quality metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            duplicates = self.df.duplicated().sum()
            st.metric(
                label="Duplicate Records",
                value=f"{duplicates:,}",
                delta=f"{(duplicates/len(self.df)*100):.2f}% of total",
                help="Number and percentage of duplicate records"
            )
        
        with col2:
            total_missing = self.df.isnull().sum().sum()
            total_cells = self.df.shape[0] * self.df.shape[1]
            completeness = (1 - total_missing / total_cells) * 100
            st.metric(
                label="Data Completeness",
                value=f"{completeness:.1f}%",
                help="Percentage of non-missing values"
            )
        
        with col3:
            unique_target = self.df[self.target].nunique()
            st.metric(
                label="Target Classes",
                value=f"{unique_target}",
                help="Number of unique values in target variable"
            )
        
        # Data types distribution
        st.subheader("📊 Data Types Distribution")
        dtype_counts = self.df.dtypes.value_counts()
        fig = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index,
            title="Data Types Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Quality insights
        st.markdown("""
        <div class="insight-box">
        <h4>🔍 Data Quality Insights:</h4>
        <ul>
        """, unsafe_allow_html=True)
        
        if len(missing_features) > 0:
            high_missing = missing_features[missing_features['Missing_Percentage'] > 30]
            if len(high_missing) > 0:
                st.markdown(f"<li>⚠️ {len(high_missing)} features have >30% missing values</li>", unsafe_allow_html=True)
            else:
                st.markdown("<li>✅ No features have excessive missing values (>30%)</li>", unsafe_allow_html=True)
        
        if duplicates > 0:
            st.markdown(f"<li>⚠️ Found {duplicates} duplicate records</li>", unsafe_allow_html=True)
        else:
            st.markdown("<li>✅ No duplicate records found</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    def show_univariate_analysis(self):
        """Display univariate analysis"""
        st.markdown('<h2 class="section-header">📊 Univariate Analysis</h2>', unsafe_allow_html=True)
        
        # Numerical features analysis
        if self.numerical_features:
            st.subheader("📈 Numerical Features Analysis")
            
            selected_features = st.multiselect(
                "Select numerical features to analyze:",
                self.numerical_features,
                default=self.numerical_features[:4] if len(self.numerical_features) >= 4 else self.numerical_features
            )
            
            if selected_features:
                # Create columns for layout
                cols = st.columns(2)
                
                for i, feature in enumerate(selected_features):
                    with cols[i % 2]:
                        # Distribution plot
                        fig = px.histogram(
                            self.df,
                            x=feature,
                            title=f'{feature} Distribution',
                            marginal="box",
                            nbins=30
                        )
                        
                        # Add statistical lines
                        mean_val = self.df[feature].mean()
                        median_val = self.df[feature].median()
                        
                        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                                    annotation_text=f"Mean: {mean_val:.2f}")
                        fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                                    annotation_text=f"Median: {median_val:.2f}")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics table
                        stats_data = {
                            'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                            'Value': [
                                f"{self.df[feature].count():,}",
                                f"{self.df[feature].mean():.4f}",
                                f"{self.df[feature].median():.4f}",
                                f"{self.df[feature].std():.4f}",
                                f"{self.df[feature].min():.4f}",
                                f"{self.df[feature].max():.4f}",
                                f"{self.df[feature].skew():.4f}",
                                f"{self.df[feature].kurtosis():.4f}"
                            ]
                        }
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Categorical features analysis
        if self.categorical_features:
            st.subheader("📊 Categorical Features Analysis")
            
            selected_cat_feature = st.selectbox(
                "Select a categorical feature to analyze:",
                self.categorical_features
            )
            
            if selected_cat_feature:
                col1, col2 = st.columns(2)
                
                # Get top categories
                value_counts = self.df[selected_cat_feature].value_counts()
                top_10 = value_counts.head(10)
                
                with col1:
                    fig = px.bar(
                        x=top_10.values,
                        y=top_10.index,
                        orientation='h',
                        title=f'{selected_cat_feature} - Top 10 Categories',
                        labels={'x': 'Count', 'y': 'Category'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Pie chart
                    if len(top_10) <= 8:
                        fig = px.pie(
                            values=top_10.values,
                            names=top_10.index,
                            title=f'{selected_cat_feature} Distribution'
                        )
                    else:
                        # Group smaller categories as "Others"
                        other_sum = value_counts.iloc[7:].sum()
                        pie_values = top_10.iloc[:7].tolist() + [other_sum]
                        pie_names = top_10.iloc[:7].index.tolist() + ['Others']
                        
                        fig = px.pie(
                            values=pie_values,
                            names=pie_names,
                            title=f'{selected_cat_feature} Distribution'
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Category statistics
                st.subheader("📋 Category Statistics")
                category_stats = pd.DataFrame({
                    'Category': value_counts.index,
                    'Count': value_counts.values,
                    'Percentage': (value_counts.values / len(self.df) * 100).round(2)
                })
                st.dataframe(category_stats.head(15), use_container_width=True, hide_index=True)
    
    def show_bivariate_analysis(self):
        """Display bivariate analysis"""
        st.markdown('<h2 class="section-header">🔗 Bivariate Analysis</h2>', unsafe_allow_html=True)
        
        # Analysis type selection
        analysis_type = st.radio(
            "Select Analysis Type:",
            ["Numerical vs Target", "Categorical vs Target"],
            horizontal=True
        )
        
        if analysis_type == "Numerical vs Target" and self.numerical_features:
            st.subheader("📈 Numerical Features vs Conversion")
            
            selected_feature = st.selectbox(
                "Select a numerical feature:",
                self.numerical_features,
                key="bivariate_num"
            )
            
            if selected_feature:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.box(
                        self.df,
                        x=self.target,
                        y=selected_feature,
                        title=f'{selected_feature} vs Conversion',
                        color=self.target.astype(str)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.violin(
                        self.df,
                        x=self.target,
                        y=selected_feature,
                        title=f'{selected_feature} Distribution by Conversion',
                        color=self.target.astype(str)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical analysis
                converted = self.df[self.df[self.target] == 1][selected_feature].dropna()
                not_converted = self.df[self.df[self.target] == 0][selected_feature].dropna()
                
                if len(converted) > 0 and len(not_converted) > 0:
                    # Mann-Whitney U test
                    try:
                        stat, p_value = stats.mannwhitneyu(converted, not_converted, alternative='two-sided')
                        
                        # Effect size (Cohen's d approximation)
                        pooled_std = np.sqrt(((len(converted) - 1) * converted.var() + (len(not_converted) - 1) * not_converted.var()) / (len(converted) + len(not_converted) - 2))
                        cohens_d = (converted.mean() - not_converted.mean()) / pooled_std if pooled_std != 0 else 0
                        
                        st.markdown(f"""
                        <div class="insight-box">
                        <h4>📊 Statistical Analysis Results:</h4>
                        <table style="width:100%">
                        <tr><td><strong>Test:</strong></td><td>Mann-Whitney U Test</td></tr>
                        <tr><td><strong>Statistic:</strong></td><td>{stat:.4f}</td></tr>
                        <tr><td><strong>P-value:</strong></td><td>{p_value:.6f}</td></tr>
                        <tr><td><strong>Significance:</strong></td><td>{'Significant ✅' if p_value < 0.05 else 'Not Significant ❌'}</td></tr>
                        <tr><td><strong>Effect Size (Cohen\'s d):</strong></td><td>{cohens_d:.4f}</td></tr>
                        </table>
                        <br>
                        <h5>Group Statistics:</h5>
                        <table style="width:100%">
                        <tr><td><strong>Converted Mean:</strong></td><td>{converted.mean():.4f}</td></tr>
                        <tr><td><strong>Not Converted Mean:</strong></td><td>{not_converted.mean():.4f}</td></tr>
                        <tr><td><strong>Difference:</strong></td><td>{converted.mean() - not_converted.mean():.4f}</td></tr>
                        <tr><td><strong>Relative Difference:</strong></td><td>{((converted.mean() - not_converted.mean()) / not_converted.mean() * 100):.2f}%</td></tr>
                        </table>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Could not perform statistical test: {str(e)}")
        
        elif analysis_type == "Categorical vs Target" and self.categorical_features:
            st.subheader("📊 Categorical Features vs Conversion")
            
            selected_feature = st.selectbox(
                "Select a categorical feature:",
                self.categorical_features,
                key="bivariate_cat"
            )
            
            if selected_feature:
                # Create crosstab
                crosstab = pd.crosstab(self.df[selected_feature], self.df[self.target])
                conversion_rates = crosstab.div(crosstab.sum(axis=1), axis=0)[1] * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Stacked bar chart
                    fig = px.bar(
                        self.df,
                        x=selected_feature,
                        color=self.target.astype(str),
                        title=f'{selected_feature} vs Conversion',
                        color_discrete_map={'0': '#ff9999', '1': '#66b3ff'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Conversion rate chart
                    fig = px.bar(
                        x=conversion_rates.index,
                        y=conversion_rates.values,
                        title=f'Conversion Rate by {selected_feature}',
                        labels={'y': 'Conversion Rate (%)', 'x': selected_feature},
                        color=conversion_rates.values,
                        color_continuous_scale='viridis'
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Chi-square test
                try:
                    chi2, p_value, dof, expected = chi2_contingency(crosstab)
                    
                    # Cramér's V (effect size)
                    n = crosstab.sum().sum()
                    cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
                    
                    st.markdown(f"""
                    <div class="insight-box">
                    <h4>📊 Chi-square Test Results:</h4>
                    <table style="width:100%">
                    <tr><td><strong>Chi-square Statistic:</strong></td><td>{chi2:.4f}</td></tr>
                    <tr><td><strong>P-value:</strong></td><td>{p_value:.6f}</td></tr>
                    <tr><td><strong>Degrees of Freedom:</strong></td><td>{dof}</td></tr>
                    <tr><td><strong>Association:</strong></td><td>{'Significant ✅' if p_value < 0.05 else 'Not Significant ❌'}</td></tr>
                    <tr><td><strong>Cramér\'s V (Effect Size):</strong></td><td>{cramers_v:.4f}</td></tr>
                    </table>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display tables
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Cross-tabulation")
                        st.dataframe(crosstab, use_container_width=True)
                    
                    with col2:
                        st.subheader("📈 Conversion Rates")
                        conv_df = pd.DataFrame({
                            'Category': conversion_rates.index,
                            'Conversion_Rate_%': conversion_rates.values.round(2)
                        }).sort_values('Conversion_Rate_%', ascending=False)
                        st.dataframe(conv_df, use_container_width=True, hide_index=True)
                        
                except Exception as e:
                    st.error(f"Could not perform chi-square test: {str(e)}")
        
        else:
            st.info("No features available for the selected analysis type.")
    
    def show_correlation_analysis(self):
        """Display correlation analysis"""
        st.markdown('<h2 class="section-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)
        
        if len(self.numerical_features) >= 2:
            # Prepare correlation matrix
            numerical_df = self.df[self.numerical_features + [self.target]].select_dtypes(include=[np.number])
            correlation_matrix = numerical_df.corr()
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Correlation heatmap
                fig = px.imshow(
                    correlation_matrix,
                    title="📊 Correlation Matrix Heatmap",
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    text_auto=True
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Target correlations
                target_corr = correlation_matrix[self.target].drop(self.target).sort_values(key=abs, ascending=False)
                
                st.subheader("🎯 Correlations with Target")
                target_corr_df = pd.DataFrame({
                    'Feature': target_corr.index,
                    'Correlation': target_corr.values.round(4)
                })
                st.dataframe(target_corr_df, use_container_width=True, hide_index=True)
                
                # Plot top correlations
                top_corr = target_corr.head(10)
                fig = px.bar(
                    x=top_corr.values,
                    y=top_corr.index,
                    orientation='h',
                    title="Top 10 Target Correlations",
                    color=top_corr.values,
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Multicollinearity check
            st.subheader("⚠️ Multicollinearity Analysis")
            
            high_corr_pairs = []
            threshold = st.slider("Correlation Threshold", 0.5, 0.9, 0.7, 0.05)
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        high_corr_pairs.append({
                            'Feature_1': correlation_matrix.columns[i],
                            'Feature_2': correlation_matrix.columns[j],
                            'Correlation': round(corr_val, 4)
                        })
            
            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs)
                st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
                
                st.markdown(f"""
                <div class="recommendation-box">
                <h4>⚠️ Multicollinearity Warning:</h4>
                <p>Found {len(high_corr_pairs)} feature pairs with correlation > {threshold}. 
                High correlations may cause multicollinearity issues in modeling.</p>
                <p><strong>Recommendations:</strong></p>
                <ul>
                <li>Consider removing one feature from each highly correlated pair</li>
                <li>Use regularization techniques (Ridge, Lasso)</li>
                <li>Apply dimensionality reduction (PCA)</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"✅ No high correlations (>{threshold}) found between features")
        
        else:
            st.info("Need at least 2 numerical features for correlation analysis")
    
    def show_feature_importance(self):
        """Display feature importance analysis"""
        st.markdown('<h2 class="section-header">⭐ Feature Importance Analysis</h2>', unsafe_allow_html=True)
        
        try:
            with st.spinner("Calculating feature importance..."):
                # Prepare data
                df_encoded = self.df.copy()
                
                # Encode categorical variables
                le = LabelEncoder()
                for feature in self.categorical_features:
                    if feature in df_encoded.columns:
                        df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str).fillna('missing'))
                
                # Prepare feature matrix
                features = [f for f in self.numerical_features + self.categorical_features if f in df_encoded.columns]
                X = df_encoded[features].fillna(0)
                y = df_encoded[self.target]
                
                # Calculate mutual information
                mi_scores = mutual_info_classif(X, y, random_state=42)
                mi_scores = pd.Series(mi_scores, index=features).sort_values(ascending=False)
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Feature importance plot
                    top_features = mi_scores.head(15)
                    fig = px.bar(
                        x=top_features.values,
                        y=top_features.index,
                        orientation='h',
                        title='🏆 Top 15 Features - Mutual Information Scores',
                        labels={'x': 'Mutual Information Score', 'y': 'Features'},
                        color=top_features.values,
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🥇 Top 10 Most Important Features")
                    importance_df = pd.DataFrame({
                        'Rank': range(1, 11),
                        'Feature': mi_scores.head(10).index,
                        'MI_Score': mi_scores.head(10).values.round(4),
                        'Type': ['Numerical' if f in self.numerical_features else 'Categorical' 
                                for f in mi_scores.head(10).index]
                    })
                    st.dataframe(importance_df, use_container_width=True, hide_index=True)
                    
                    # Feature type breakdown
                    top_10_features = mi_scores.head(10).index
                    top_numerical = sum(1 for f in top_10_features if f in self.numerical_features)
                    top_categorical = sum(1 for f in top_10_features if f in self.categorical_features)
                    
                    st.metric("🔢 Numerical in Top 10", top_numerical)
                    st.metric("📊 Categorical in Top 10", top_categorical)
                
                # Complete ranking
                st.subheader("📋 Complete Feature Importance Ranking")
                
                # Filter for display
                score_threshold = st.slider("Minimum MI Score", 0.0, mi_scores.max(), 0.0, 0.01)
                filtered_scores = mi_scores[mi_scores >= score_threshold]
                
                all_importance_df = pd.DataFrame({
                    'Rank': range(1, len(filtered_scores) + 1),
                    'Feature': filtered_scores.index,
                    'MI_Score': filtered_scores.values.round(4),
                    'Type': ['Numerical' if f in self.numerical_features else 'Categorical' 
                            for f in filtered_scores.index]
                })
                st.dataframe(all_importance_df, use_container_width=True, hide_index=True)
                
                # Insights
                high_importance_features = mi_scores[mi_scores > 0.1]
                medium_importance_features = mi_scores[(mi_scores > 0.05) & (mi_scores <= 0.1)]
                
                st.markdown(f"""
                <div class="insight-box">
                <h4>🔍 Feature Importance Insights:</h4>
                <ul>
                <li>🏆 <strong>Top performing feature:</strong> {mi_scores.index[0]} (Score: {mi_scores.iloc[0]:.4f})</li>
                <li>🔥 <strong>High importance features (>0.1):</strong> {len(high_importance_features)}</li>
                <li>📊 <strong>Medium importance features (0.05-0.1):</strong> {len(medium_importance_features)}</li>
                <li>📈 <strong>Numerical features in top 10:</strong> {top_numerical}</li>
                <li>📋 <strong>Categorical features in top 10:</strong> {top_categorical}</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error calculating feature importance: {str(e)}")
            st.info("This might occur due to data formatting issues. Please check your dataset.")
    
    def show_business_insights(self):
        """Display business-specific insights"""
        st.markdown('<h2 class="section-header">💼 Business Insights</h2>', unsafe_allow_html=True)
        
        # Lead source analysis
        if 'Lead Source' in self.df.columns:
            st.subheader("📈 Lead Source Performance Analysis")
            
            lead_stats = self.df.groupby('Lead Source').agg({
                self.target: ['count', 'sum', 'mean']
            }).round(4)
            
            lead_stats.columns = ['Total_Leads', 'Conversions', 'Conversion_Rate']
            lead_stats = lead_stats.sort_values('Conversion_Rate', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    lead_stats,
                    x=lead_stats.index,
                    y='Total_Leads',
                    title='📊 Total Leads by Source',
                    color='Total_Leads',
                    color_continuous_scale='blues'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    lead_stats,
                    x=lead_stats.index,
                    y='Conversion_Rate',
                    title='🎯 Conversion Rate by Lead Source',
                    color='Conversion_Rate',
                    color_continuous_scale='viridis'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Lead source table
            st.dataframe(lead_stats, use_container_width=True)
            
            # Performance insights
            best_source = lead_stats.index[0]
            worst_source = lead_stats.index[-1]
            best_rate = lead_stats.loc[best_source, 'Conversion_Rate']
            worst_rate = lead_stats.loc[worst_source, 'Conversion_Rate']
            
            st.markdown(f"""
            <div class="insight-box">
            <h4>📊 Lead Source Performance Insights:</h4>
            <ul>
            <li>🏆 <strong>Best performing source:</strong> {best_source} ({best_rate:.2%} conversion rate)</li>
            <li>📉 <strong>Lowest performing source:</strong> {worst_source} ({worst_rate:.2%} conversion rate)</li>
            <li>📈 <strong>Performance gap:</strong> {(best_rate - worst_rate):.2%}</li>
            <li>💡 <strong>Recommendation:</strong> Focus marketing budget on top-performing sources</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Website engagement analysis
        website_features = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
        existing_website_features = [f for f in website_features if f in self.df.columns]
        
        if existing_website_features:
            st.subheader("🌐 Website Engagement Analysis")
            
            engagement_data = []
            for feature in existing_website_features:
                converted_avg = self.df[self.df[self.target] == 1][feature].mean()
                not_converted_avg = self.df[self.df[self.target] == 0][feature].mean()
                difference_pct = ((converted_avg - not_converted_avg) / not_converted_avg * 100) if not_converted_avg != 0 else 0
                
                engagement_data.append({
                    'Metric': feature,
                    'Converted_Avg': round(converted_avg, 2),
                    'Not_Converted_Avg': round(not_converted_avg, 2),
                    'Difference_%': round(difference_pct, 2)
                })
            
            engagement_df = pd.DataFrame(engagement_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    engagement_df,
                    x='Metric',
                    y=['Converted_Avg', 'Not_Converted_Avg'],
                    title='📊 Website Engagement: Converted vs Not Converted',
                    barmode='group',
                    color_discrete_sequence=['#66b3ff', '#ff9999']
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    engagement_df,
                    x='Metric',
                    y='Difference_%',
                    title='📈 Percentage Difference in Engagement',
                    color='Difference_%',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(engagement_df, use_container_width=True, hide_index=True)
        
        # Activity-based insights
        if 'Last Activity' in self.df.columns:
            st.subheader("🎯 Last Activity Performance Analysis")
            
            activity_stats = self.df.groupby('Last Activity').agg({
                self.target: ['count', 'mean']
            }).round(4)
            
            activity_stats.columns = ['Count', 'Conversion_Rate']
            activity_stats = activity_stats.sort_values('Conversion_Rate', ascending=False)
            
            # Filter activities with sufficient sample size
            min_count = st.slider("Minimum activity count", 1, 50, 10)
            filtered_activities = activity_stats[activity_stats['Count'] >= min_count]
            
            if len(filtered_activities) > 0:
                fig = px.bar(
                    filtered_activities.head(10),
                    x=filtered_activities.head(10).index,
                    y='Conversion_Rate',
                    title=f'🏆 Top 10 Activities by Conversion Rate (min {min_count} occurrences)',
                    color='Conversion_Rate',
                    color_continuous_scale='viridis'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(filtered_activities.head(15), use_container_width=True)
        
        # ROI and Business Impact Analysis
        st.subheader("💰 Business Impact Analysis")
        
        # Calculate key business metrics
        total_leads = len(self.df)
        total_conversions = self.df[self.target].sum()
        conversion_rate = total_conversions / total_leads
        
        # Hypothetical values for demonstration
        avg_deal_value = st.number_input("Average Deal Value ($)", value=1000, min_value=1)
        cost_per_lead = st.number_input("Cost per Lead ($)", value=50, min_value=1)
        
        if st.button("Calculate Business Impact"):
            total_revenue = total_conversions * avg_deal_value
            total_cost = total_leads * cost_per_lead
            roi = ((total_revenue - total_cost) / total_cost) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
            
            with col2:
                st.metric("💸 Total Cost", f"${total_cost:,.0f}")
            
            with col3:
                st.metric("📈 ROI", f"{roi:.1f}%")
            
            with col4:
                st.metric("💵 Revenue per Lead", f"${total_revenue/total_leads:.0f}")
    
    def show_outlier_analysis(self):
        """Display outlier analysis"""
        st.markdown('<h2 class="section-header">🎯 Outlier Analysis</h2>', unsafe_allow_html=True)
        
        if self.numerical_features:
            # Calculate outlier statistics
            outlier_data = []
            
            for feature in self.numerical_features:
                Q1 = self.df[feature].quantile(0.25)
                Q3 = self.df[feature].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
                outlier_percentage = (len(outliers) / len(self.df)) * 100
                
                outlier_data.append({
                    'Feature': feature,
                    'Outlier_Count': len(outliers),
                    'Outlier_Percentage': round(outlier_percentage, 2),
                    'Lower_Bound': round(lower_bound, 4),
                    'Upper_Bound': round(upper_bound, 4),
                    'Q1': round(Q1, 4),
                    'Q3': round(Q3, 4),
                    'IQR': round(IQR, 4)
                })
            
            outlier_df = pd.DataFrame(outlier_data).sort_values('Outlier_Percentage', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Outlier Summary")
                st.dataframe(outlier_df[['Feature', 'Outlier_Count', 'Outlier_Percentage']], 
                           use_container_width=True, hide_index=True)
            
            with col2:
                fig = px.bar(
                    outlier_df.head(10),
                    x='Feature',
                    y='Outlier_Percentage',
                    title='📈 Outlier Percentage by Feature',
                    color='Outlier_Percentage',
                    color_continuous_scale='reds'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            # Interactive outlier visualization
            st.subheader("🔍 Interactive Outlier Visualization")
            
            selected_feature = st.selectbox(
                "Select a feature for detailed outlier analysis:",
                self.numerical_features
            )
            
            if selected_feature:
                feature_data = outlier_df[outlier_df['Feature'] == selected_feature].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.box(
                        self.df,
                        y=selected_feature,
                        title=f'📦 {selected_feature} - Box Plot with Outliers'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(
                        self.df,
                        x=selected_feature,
                        title=f'📊 {selected_feature} - Distribution',
                        marginal="box",
                        nbins=50
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Outlier details
                st.markdown(f"""
                <div class="insight-box">
                <h4>📊 {selected_feature} Outlier Analysis:</h4>
                <table style="width:100%">
                <tr><td><strong>Total outliers:</strong></td><td>{feature_data['Outlier_Count']}</td></tr>
                <tr><td><strong>Outlier percentage:</strong></td><td>{feature_data['Outlier_Percentage']:.2f}%</td></tr>
                <tr><td><strong>Lower bound:</strong></td><td>{feature_data['Lower_Bound']:.4f}</td></tr>
                <tr><td><strong>Upper bound:</strong></td><td>{feature_data['Upper_Bound']:.4f}</td></tr>
                <tr><td><strong>Q1:</strong></td><td>{feature_data['Q1']:.4f}</td></tr>
                <tr><td><strong>Q3:</strong></td><td>{feature_data['Q3']:.4f}</td></tr>
                <tr><td><strong>IQR:</strong></td><td>{feature_data['IQR']:.4f}</td></tr>
                </table>
                </div>
                """, unsafe_allow_html=True)
                
                # Outlier treatment recommendations
                if feature_data['Outlier_Percentage'] > 5:
                    st.markdown("""
                    <div class="recommendation-box">
                    <h4>⚠️ High Outlier Percentage Detected!</h4>
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                    <li>🔍 Investigate outliers for data entry errors</li>
                    <li>📊 Consider winsorization (cap at 95th/5th percentile)</li>
                    <li>🔄 Apply log transformation if appropriate</li>
                    <li>🎯 Use robust models (Random Forest, etc.)</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ Outlier percentage is within acceptable range (<5%)")
        
        else:
            st.info("No numerical features available for outlier analysis")
    
    def show_final_report(self):
        """Display comprehensive final report"""
        st.markdown('<h2 class="section-header">📋 Executive Summary & Recommendations</h2>', unsafe_allow_html=True)
        
        # Key metrics dashboard
        conversion_rate = (self.df[self.target].sum() / len(self.df)) * 100
        missing_pct = (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 Dataset Size",
                value=f"{self.df.shape[0]:,}",
                help="Total number of records analyzed"
            )
        
        with col2:
            st.metric(
                label="🔢 Features",
                value=f"{self.df.shape[1]:,}",
                help="Total number of features"
            )
        
        with col3:
            st.metric(
                label="🎯 Conversion Rate",
                value=f"{conversion_rate:.2f}%",
                help="Overall conversion rate"
            )
        
        with col4:
            st.metric(
                label="✅ Data Quality",
                value=f"{100-missing_pct:.1f}%",
                help="Data completeness percentage"
            )
        
        # Executive insights
        st.markdown("""
        <div class="insight-box">
        <h3>🔍 Key Business Insights</h3>
        """, unsafe_allow_html=True)
        
        insights = []
        
        # Conversion rate assessment
        if conversion_rate < 10:
            insights.append("⚠️ **Low conversion rate** - Significant optimization opportunities exist")
        elif conversion_rate > 30:
            insights.append("✅ **High conversion rate** - Effective lead generation and nurturing process")
        else:
            insights.append("📊 **Moderate conversion rate** - Room for improvement with targeted optimization")
        
        # Data quality assessment
        if missing_pct < 5:
            insights.append("✅ **Excellent data quality** - Minimal missing values")
        elif missing_pct < 15:
            insights.append("📊 **Good data quality** - Some missing values to address")
        else:
            insights.append("⚠️ **Data quality concerns** - Significant missing values require attention")
        
        # Feature insights
        if len(self.numerical_features) > len(self.categorical_features):
            insights.append(f"📈 **Quantitative focus** - {len(self.numerical_features)} numerical vs {len(self.categorical_features)} categorical features")
        else:
            insights.append(f"📊 **Categorical focus** - {len(self.categorical_features)} categorical vs {len(self.numerical_features)} numerical features")
        
        for insight in insights:
            st.markdown(f"<li>{insight}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Technical recommendations
        st.markdown("""
        <div class="recommendation-box">
        <h3>🛠️ Technical Recommendations</h3>
        """, unsafe_allow_html=True)
        
        recommendations = []
        
        # Data preprocessing
        if missing_pct > 10:
            recommendations.append("🔧 **Data Cleaning**: Address missing values through imputation or feature removal")
        
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            recommendations.append(f"🔍 **Duplicate Handling**: Remove {duplicates} duplicate records")
        
        # Class imbalance
        if conversion_rate < 20:
            recommendations.append("⚖️ **Class Imbalance**: Use stratified sampling, SMOTE, or cost-sensitive learning")
        
        # Feature engineering
        recommendations.append("🔧 **Feature Engineering**: Create interaction features and polynomial terms")
        recommendations.append("📊 **Feature Selection**: Use top features from importance analysis")
        
        # Model recommendations
        if len(self.numerical_features) + len(self.categorical_features) > 10:
            recommendations.append("🤖 **Model Strategy**: Consider ensemble methods (Random Forest, XGBoost)")
        else:
            recommendations.append("🤖 **Model Strategy**: Start with logistic regression and tree-based models")
        
        recommendations.append("🎯 **Evaluation Metrics**: Focus on precision, recall, and F1-score")
        recommendations.append("✅ **Validation**: Use stratified k-fold cross-validation")
        
        for rec in recommendations:
            st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Modeling roadmap
        st.subheader("🚀 Recommended Modeling Roadmap")
        
        roadmap_steps = [
            ("1️⃣", "**Data Preprocessing**", "Handle missing values, encode categories, scale features"),
            ("2️⃣", "**Feature Engineering**", "Create new features, handle outliers, feature selection"),
            ("3️⃣", "**Model Development**", "Start with baseline models, then ensemble methods"),
            ("4️⃣", "**Model Validation**", "Cross-validation with business-relevant metrics"),
            ("5️⃣", "**Hyperparameter Tuning**", "Optimize model performance systematically"),
            ("6️⃣", "**Model Interpretation**", "SHAP values, feature importance, business insights"),
            ("7️⃣", "**Production Deployment**", "API development, monitoring, and maintenance"),
            ("8️⃣", "**Continuous Improvement**", "A/B testing, model retraining, performance monitoring")
        ]
        
        for step_num, title, description in roadmap_steps:
            st.markdown(f"**{step_num} {title}**: {description}")
        
        # Download report functionality
        if st.button("📄 Generate Detailed Report Summary"):
            report_data = {
                "Executive_Summary": {
                    "dataset_size": self.df.shape[0],
                    "features_count": self.df.shape[1],
                    "conversion_rate": f"{conversion_rate:.2f}%",
                    "data_quality": f"{100-missing_pct:.1f}%"
                },
                "Feature_Analysis": {
                    "numerical_features": len(self.numerical_features),
                    "categorical_features": len(self.categorical_features),
                    "target_variable": self.target
                },
                "Key_Recommendations": recommendations[:5]  # Top 5 recommendations
            }
            
            st.json(report_data)
            st.success("✅ Report summary generated! Copy the JSON above for documentation.")

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">🎯 Sales Conversion Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize the dashboard
    dashboard = SalesConversionDashboard()
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation Panel")
    st.sidebar.markdown("---")
    
    # Data loading section
    if dashboard.load_data():
        dashboard.categorize_features()
        
        # Analysis navigation
        analysis_sections = {
            "📈 Dataset Overview": "overview",
            "🔍 Data Quality": "quality", 
            "📊 Univariate Analysis": "univariate",
            "🔗 Bivariate Analysis": "bivariate",
            "🔗 Correlation Analysis": "correlation",
            "⭐ Feature Importance": "importance",
            "💼 Business Insights": "business",
            "🎯 Outlier Analysis": "outliers",
            "📋 Executive Summary": "summary"
        }
        
        selected_section = st.sidebar.selectbox(
            "Choose Analysis Section:",
            list(analysis_sections.keys())
        )
        
        # Display selected analysis
        section_key = analysis_sections[selected_section]
        
        if section_key == "overview":
            dashboard.show_overview()
        elif section_key == "quality":
            dashboard.show_data_quality()
        elif section_key == "univariate":
            dashboard.show_univariate_analysis()
        elif section_key == "bivariate":
            dashboard.show_bivariate_analysis()
        elif section_key == "correlation":
            dashboard.show_correlation_analysis()
        elif section_key == "importance":
            dashboard.show_feature_importance()
        elif section_key == "business":
            dashboard.show_business_insights()
        elif section_key == "outliers":
            dashboard.show_outlier_analysis()
        elif section_key == "summary":
            dashboard.show_final_report()
        
        # Sidebar quick stats
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Quick Statistics")
        
        conversion_rate = (dashboard.df[dashboard.target].sum() / len(dashboard.df)) * 100
        st.sidebar.metric("Records", f"{dashboard.df.shape[0]:,}")
        st.sidebar.metric("Features", f"{dashboard.df.shape[1]:,}")
        st.sidebar.metric("Conversion Rate", f"{conversion_rate:.1f}%")
        
        # Sidebar tips
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💡 Analysis Tips")
        st.sidebar.info("""
        🔍 **Start Here**: Dataset Overview → Data Quality
        
        📊 **Core Analysis**: Univariate → Bivariate → Correlation
        
        🎯 **Advanced**: Feature Importance → Business Insights
        
        📋 **Summary**: Executive Summary for key findings
        """)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ About This Dashboard")
        st.sidebar.markdown("""
        **Version**: 1.0  
        **Purpose**: Sales Conversion EDA  
        **Features**: Interactive analysis with business insights
        """)
    
    else:
        # Welcome screen when no data is loaded
        st.markdown("""
        ## 🚀 Welcome to the Sales Conversion Analysis Dashboard
        
        This comprehensive dashboard provides **professional-grade exploratory data analysis** 
        specifically designed for sales conversion prediction datasets.
        
        ### 📋 What You'll Get:
        
        #### 🔍 **Comprehensive Analysis**
        - **Data Quality Assessment** - Missing values, duplicates, data types
        - **Statistical Analysis** - Distributions, correlations, significance tests
        - **Feature Importance** - Mutual information scoring and ranking
        - **Business Insights** - Lead source performance, ROI analysis
        - **Visual Analytics** - Interactive Plotly charts and visualizations
        
        #### 🎯 **Business-Focused Insights**
        - Lead source performance comparison
        - Website engagement impact analysis
        - Conversion rate optimization recommendations
        - ROI and business impact calculations
        
        #### 🛠️ **Technical Excellence**
        - Statistical significance testing
        - Outlier detection and analysis
        - Multicollinearity assessment
        - Feature engineering recommendations
        
        ### 📁 **Getting Started**
        
        1. **Upload your CSV file** using the file uploader above
        2. **Ensure your dataset includes**:
           - A target column named `'Converted'` with 0/1 values
           - Various feature columns (numerical and categorical)
           - Optional: `'Lead Source'`, `'Last Activity'`, website metrics
        
        3. **Navigate through sections** using the sidebar menu
        4. **Generate insights** and recommendations for your business
        
        ### 🎨 **Dashboard Features**
        
        ✅ **Interactive Visualizations** - Plotly charts with hover details  
        ✅ **Statistical Testing** - Chi-square, Mann-Whitney U tests  
        ✅ **Professional Styling** - Clean, modern interface  
        ✅ **Export Capabilities** - JSON report generation  
        ✅ **Mobile Responsive** - Works on all devices  
        ✅ **Real-time Analysis** - Instant results as you explore  
        
        ### 📊 **Perfect For**
        - Sales teams analyzing lead conversion
        - Marketing teams optimizing campaigns  
        - Data scientists building prediction models
        - Business analysts seeking actionable insights
        
        ---
        
        **Ready to analyze your data?** Upload your CSV file above to begin! 🚀
        """)

if __name__ == "__main__":
    main()