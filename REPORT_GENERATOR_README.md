# 📊 SHAP Global Explanations Report Generator

A powerful standalone tool that automatically generates beautiful HTML reports with SHAP Global Explanations for bias analysis. This tool works independently or alongside the main bias detection system to provide detailed feature importance analysis.

## 🎯 **Overview**

The Report Generator creates professional, standalone HTML reports showing how demographic features influence model predictions using SHAP (SHapley Additive exPlanations) values. Perfect for bias analysis, model interpretability, and stakeholder presentations.

## 🚀 **Key Features**

### **🤖 SHAP-Based Analysis**
- **True SHAP Values**: Uses SHAP TreeExplainer with Random Forest
- **Per-Class Analysis**: Separate feature importance for each target class
- **Feature Ranking**: Top features that influence predictions
- **Importance Percentages**: Relative contribution of each feature

### **📋 Automated Processing**
- **Smart Feature Selection**: Focuses on demographic attributes
- **Automatic Detection**: Monitors uploads directory for new files
- **Error Handling**: Graceful fallback to feature importance
- **Sample Optimization**: Uses 500 samples for efficient SHAP calculation

### **🎨 Professional Reports**
- **Bootstrap Styling**: Modern, responsive design
- **Color-Coded Importance**: Visual indicators for feature importance
- **Standalone HTML**: No dependencies, works offline
- **Print-Ready**: Perfect for documentation and presentations

## 🏗️ **How It Works**

```
📁 CSV Dataset Upload
    │
    ▼
🔍 LLM Column Analysis
    │
    ▼
🎯 Feature Selection (Demographic Only)
    │
    ▼
🤖 SHAP Calculation
    │
    ▼
📊 Report Generation
    │
    ▼
🌐 HTML Report (Standalone)
```

## 🛠️ **Installation**

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Configure LLM Endpoints** (if using LLM analysis)
Edit `app/config.py` with your API keys:
```python
LLM_ENDPOINTS = {
    'llama_3_3': {
        'url': 'https://api.groq.com/openai/v1/chat/completions',
        'headers': {'Authorization': 'Bearer YOUR_API_KEY'}
    }
}
```

## 📖 **Usage Guide**

### **Option 1: Manual Report Generation**

Generate a report for a specific file:

```bash
# Basic usage
python generate_global_explanations_report.py your_dataset.csv

# With options
python generate_global_explanations_report.py your_dataset.csv --n-rows 1000 --test-size 0.2
```

**Available Options:**
- `--n-rows 1000`: Limit analysis to first 1000 rows
- `--test-size 0.2`: Set test size for model evaluation (default: 0.2)
- `--max-categories 50`: Maximum categories for categorical features
- `--output-dir reports`: Set output directory for reports

### **Option 2: Automatic Monitoring**

Start the monitoring service to automatically generate reports:

```bash
python auto_generate_report.py
```

**What happens:**
1. Monitors the `uploads/` directory for new CSV files
2. Automatically generates SHAP reports when files are uploaded
3. Opens reports in your default browser
4. Saves reports with timestamps

## 📊 **Report Content**

### **📋 Dataset Information**
- **File Details**: Name, rows, columns
- **Target Column**: Automatically detected by LLM
- **Protected Attributes**: Demographic features analyzed
- **Generation Timestamp**: When the report was created

### **📈 SHAP Global Explanations**
- **Per-Class Tables**: Separate analysis for each target class
- **Feature Rankings**: Top 10 most important features per class
- **SHAP Values**: Actual SHAP importance scores
- **Importance Percentages**: Relative contribution of each feature
- **Color Coding**:
  - 🟢 **High Importance** (>15%): Heavily influences predictions
  - 🟡 **Medium Importance** (8-15%): Moderate influence
  - ⚪ **Low Importance** (<8%): Minimal influence

### **🎨 Visual Design**
- **Professional Layout**: Bootstrap-based responsive design
- **Color-Coded Rows**: Visual importance indicators
- **Badges and Icons**: Clear visual hierarchy
- **Print-Friendly**: Optimized for printing and PDF export

## 📁 **File Structure**

```
reports/
├── shap_global_explanations_report_20241201_143022.html
├── shap_global_explanations_report_20241201_143156.html
└── ...

uploads/
├── your_dataset.csv
└── another_dataset.csv
```

## 🔧 **Configuration**

### **Feature Selection**
The system focuses on these demographic features:
```python
allowed_features = [
    'SUSP_RACE', 'SUSP_SEX', 'SUSP_AGE_GROUP',
    'VIC_RACE', 'VIC_SEX', 'VIC_AGE_GROUP'
]
```

### **SHAP Settings**
- **Sample Size**: 500 samples (configurable)
- **Model**: Random Forest with 100 estimators
- **Explainer**: SHAP TreeExplainer
- **Fallback**: Feature importance if SHAP fails

## 📊 **Example Output**

### **Report Header**
```
SHAP Global Explanations Report
SHAP Feature Importance Analysis for Machine Learning Model
[Powered by SHAP]
```

### **Dataset Summary**
```
Dataset Information:
- Filename: NYPD_Complaint_Data_Historic_20250515_preprocessed.csv
- Total Rows: 10,000
- Total Columns: 35
- Target Column: LAW_CAT_CD
```

### **Feature Importance Table**
```
Rank | Feature        | SHAP Importance | Percentage
-----|----------------|-----------------|------------
1    | SUSP_RACE      | 0.2345         | 23.5%
2    | SUSP_SEX       | 0.1892         | 18.9%
3    | VIC_RACE       | 0.1567         | 15.7%
...
```

## 🔍 **Supported Datasets**

### **Law Enforcement Data**
- **NYPD Crime Data**: Tested and optimized
- **LAPD Crime Data**: Compatible format
- **Any Crime Dataset**: With demographic columns

### **Required Columns**
- **Demographic Features**: Race, sex, age columns
- **Target Variable**: Crime categories, outcomes, etc.
- **Format**: CSV with headers

## 🆘 **Troubleshooting**

### **Common Issues**

**SHAP Calculation Fails**
```
❌ Error: Per-column arrays must each be 1-dimensional
```
**Solution**: System automatically falls back to feature importance

**Memory Issues**
```
❌ Error: Out of memory
```
**Solution**: Use `--n-rows 1000` to limit dataset size

**LLM API Errors**
```
❌ Error: API key invalid
```
**Solution**: Check API keys in `app/config.py`

### **Performance Tips**

**Large Datasets**
```bash
# Limit rows for faster processing
python generate_global_explanations_report.py large_file.csv --n-rows 5000
```

**Multiple Reports**
```bash
# Start monitoring for automatic generation
python auto_generate_report.py
```

**Report Customization**
- Edit `allowed_features` in the script for different feature sets
- Modify HTML templates for custom styling
- Adjust SHAP sample size for speed vs. accuracy trade-off

## 🔗 **Integration with Main System**

### **Workflow**
1. **Main App**: Upload files and get bias analysis
2. **Report Generator**: Automatically creates SHAP reports
3. **Both Systems**: Use the same uploads directory

### **Benefits**
- **Complementary Analysis**: Web interface + standalone reports
- **Different Audiences**: Interactive vs. documentation
- **Flexible Usage**: Use independently or together

## 📈 **Advanced Usage**

### **Custom Feature Sets**
Edit the `allowed_features` list in the script:
```python
allowed_features = [
    'YOUR_RACE_COL', 'YOUR_SEX_COL', 'YOUR_AGE_COL',
    # Add your specific demographic columns
]
```

### **Batch Processing**
```bash
# Process multiple files
for file in uploads/*.csv; do
    python generate_global_explanations_report.py "$file"
done
```

### **Scheduled Reports**
```bash
# Run daily at 9 AM
0 9 * * * cd /path/to/dsai2 && python auto_generate_report.py
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Add new report features or styling
4. Test with different datasets
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License.

---

**📊 Generate professional SHAP reports for bias analysis and model interpretability!** 
 