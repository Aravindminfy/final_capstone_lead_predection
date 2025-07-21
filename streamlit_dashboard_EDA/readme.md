'''

# Streamlit Sales Conversion Analysis Dashboard  
**File:** streamlit_dash.py  
**Purpose:** Interactive, production-grade EDA and business insights for sales conversion datasets.

---

## 1. **Overview & Flow**

This dashboard is designed for **exploratory data analysis (EDA)** and **business insight generation** on sales conversion datasets. It guides users from data upload through detailed statistical analysis, feature importance, business impact, and actionable recommendations.

**Main Flow:**
1. **User uploads CSV dataset** (must include a `Converted` column).
2. **Dashboard validates and loads data**.
3. **Sidebar navigation** lets users select analysis sections:
   - Dataset Overview
   - Data Quality
   - Univariate Analysis
   - Bivariate Analysis
   - Correlation Analysis
   - Feature Importance
   - Business Insights
   - Outlier Analysis
   - Executive Summary
4. **Each section provides interactive visualizations, metrics, and insights**.
5. **Final report and recommendations** are generated for business and technical teams.

---

## 2. **Key Components & Logic**

### **Imports & Setup**
- Uses **Streamlit** for UI, **Pandas/Numpy** for data, **Plotly** for interactive charts, **Seaborn/Matplotlib** for additional plotting, **Scipy/Sklearn** for statistics and feature selection.
- **Custom CSS** enhances visual styling for clarity and professionalism.

### **Data Loading**
- **File uploader** accepts CSV files.
- **Validation:** Checks for required `Converted` column.
- **Caching:** Uses `@st.cache_data` for efficient reloads.
- **Column info:** Displays data types, null counts, and non-null counts.

### **Feature Categorization**
- Automatically separates **numerical** and **categorical** features.
- Removes common ID columns and the target from analysis features.

### **Analysis Sections**

#### **Dataset Overview**
- **Key metrics:** Total records, features, conversion rate, conversions.
- **Target distribution:** Pie and bar charts.
- **Sample data:** Shows first 10 rows.
- **Feature lists:** Numerical and categorical features.

#### **Data Quality**
- **Missing values:** Table and bar chart for top missing features.
- **Duplicates:** Count and percentage.
- **Completeness:** Overall data completeness metric.
- **Data types:** Pie chart of types.
- **Insights:** Flags high missing/duplicate features.

#### **Univariate Analysis**
- **Numerical:** Histograms, box plots, and statistics (mean, median, std, skewness, kurtosis).
- **Categorical:** Bar and pie charts for top categories, category stats.

#### **Bivariate Analysis**
- **Numerical vs Target:** Box/violin plots, Mann-Whitney U test, Cohen’s d effect size, group means.
- **Categorical vs Target:** Stacked bar, conversion rate chart, chi-square test, Cramér’s V, crosstab.

#### **Correlation Analysis**
- **Correlation matrix:** Heatmap for numerical features.
- **Target correlations:** Top features correlated with conversion.
- **Multicollinearity:** Detects highly correlated pairs, recommends mitigation.

#### **Feature Importance**
- **Mutual Information:** Ranks features by predictive power.
- **Top features:** Bar chart, breakdown by type.
- **Complete ranking:** Filterable by score.
- **Insights:** Highlights top features and their types.

#### **Business Insights**
- **Lead Source:** Performance by source, conversion rates, recommendations.
- **Website Engagement:** Compares engagement metrics for converted vs not converted.
- **Last Activity:** Conversion rates by activity, filterable by sample size.
- **ROI Analysis:** Calculates revenue, cost, ROI, revenue per lead based on user inputs.

#### **Outlier Analysis**
- **IQR-based detection:** Outlier count and percentage per feature.
- **Visualization:** Box and histogram for selected feature.
- **Recommendations:** Actions for high outlier features.

#### **Executive Summary**
- **Key metrics:** Dataset size, features, conversion rate, data quality.
- **Business insights:** Conversion rate assessment, data quality, feature focus.
- **Technical recommendations:** Data cleaning, duplicate handling, class imbalance, feature engineering, model strategy.
- **Modeling roadmap:** Step-by-step guide for production ML.
- **Report export:** JSON summary for documentation.

---

## 3. **Production-Level Reasoning**

- **Automated feature detection** ensures adaptability to different datasets.
- **Statistical tests** (Mann-Whitney U, chi-square) provide rigorous, unbiased insights.
- **Mutual information** for feature importance is model-agnostic and robust.
- **Business metrics** (ROI, conversion rate) connect data analysis to actionable business decisions.
- **Multicollinearity and outlier checks** prevent common modeling pitfalls.
- **Recommendations and roadmap** guide users toward best practices for ML deployment.
- **Caching and efficient data handling** ensure scalability for large datasets.
- **Professional styling and interactive charts** improve usability and stakeholder communication.

---

## 4. **Why This Approach?**

- **End-to-end EDA:** Covers all critical steps from data quality to business impact.
- **User-centric:** Interactive, easy to navigate, and actionable.
- **Scalable:** Handles large datasets and complex feature sets.
- **Reproducible:** Caching, clear logic, and exportable reports.
- **Business-aligned:** Directly supports sales and marketing optimization.

---

## 5. **How to Use**

1. **Upload your CSV file** (must have a `Converted` column).
2. **Navigate sections** via sidebar.
3. **Interact with charts and tables** for deeper insights.
4. **Review recommendations and roadmap** for next steps.
5. **Export summary** for documentation or reporting.

---

## 6. **Extensibility**

- Easily add new analysis modules (e.g., time series, segmentation).
- Integrate with ML model training and prediction.
- Connect to databases or cloud storage for automated data refresh.

---

## 7. **Summary**

This dashboard provides a **comprehensive, production-ready EDA and business insight platform** for sales conversion datasets. It combines statistical rigor, business relevance, and technical best practices to empower data-driven decision making.

---


'''
<img width="1909" height="930" alt="image" src="https://github.com/user-attachments/assets/c6639f15-b6d0-428c-8219-457689b3948f" />
<img width="1916" height="924" alt="image" src="https://github.com/user-attachments/assets/c11dcfe1-a637-4c36-bb7f-a77fa6259fa1" />
<img width="1919" height="930" alt="image" src="https://github.com/user-attachments/assets/17809e2d-1ead-4bec-bcb1-afbb3087d6f9" />
<img width="1918" height="931" alt="image" src="https://github.com/user-attachments/assets/71ae44a1-f1a7-4ccb-8f92-c52b97f351f2" />
<img width="1911" height="914" alt="image" src="https://github.com/user-attachments/assets/35540fdc-8fbe-4cd5-a6ef-ed8ce2708423" />
<img width="1906" height="889" alt="image" src="https://github.com/user-attachments/assets/3e80fe19-2a18-4299-93ee-b45227c7ea98" />
<img width="1908" height="919" alt="image" src="https://github.com/user-attachments/assets/3e3bf020-9632-4001-b61c-ebb1d3673588" />
<img width="1915" height="933" alt="image" src="https://github.com/user-attachments/assets/406e1886-1556-4249-8691-8ad375c6ee56" />



















