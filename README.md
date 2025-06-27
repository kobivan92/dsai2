# 🤖 Multi-LLM Bias Detection & Analysis System

**GitHub Repository**: [https://github.com/kobivan92/dsai2](https://github.com/kobivan92/dsai2)

A comprehensive bias detection and analysis platform that uses multiple Large Language Models (LLMs) to identify, analyze, and report on biases in datasets. The system combines AIF360 fairness metrics, SHAP explanations, and multi-LLM insights for robust bias detection.

## 🎯 **System Overview**

This system provides **automated bias detection** using three different LLMs (llama_3_3, deepseek_r1, mistral_nemo) to analyze datasets for fairness issues across protected attributes like race, gender, and age.

## 🏗️ **Architecture & Pipeline Schema**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BIAS DETECTION PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────┘

📁 INPUT DATASET (CSV)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔍 STEP 1: LLM COLUMN ANALYSIS                                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │   llama_3_3     │ │   deepseek_r1   │ │   mistral_nemo  │              │
│  │                 │ │                 │ │                 │              │
│  │ • Target Column │ │ • Target Column │ │ • Target Column │              │
│  │ • Protected Attr│ │ • Protected Attr│ │ • Protected Attr│              │
│  │ • Excluded Cols │ │ • Excluded Cols │ │ • Excluded Cols │              │
│  │ • Race Column   │ │ • Race Column   │ │ • Race Column   │              │
│  │ • Privileged    │ │ • Privileged    │ │ • Privileged    │              │
│  │ • Unprivileged  │ │ • Unprivileged  │ │ • Unprivileged  │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 STEP 2: DATA PREPROCESSING & MODEL TRAINING                            │
│  • Drop excluded columns                                                    │
│  • Handle missing values                                                    │
│  • Encode categorical features                                              │
│  • Train Random Forest classifier                                           │
│  • Calculate performance metrics                                            │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  📊 STEP 3: SHAP GLOBAL EXPLANATIONS                                       │
│  • Calculate SHAP values for each class                                     │
│  • Feature importance ranking                                               │
│  • Per-class analysis                 
│  • Generate standalone HTML reports                                         │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ⚖️ STEP 4: AIF360 FAIRNESS METRICS                                        │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                │
│  │ Statistical     │ │ Disparate       │ │ Mean            │                │
│  │ Parity          │ │ Impact          │ │ Difference      │                │
│  │ Difference      │ │                 │ │                 │                │
│  │                 │ │                 │ │                 │                │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  🎯 STEP 5: BIAS LEVEL CLASSIFICATION                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SCORING SYSTEM (0-9 points)                     │   │
│  │                                                                     │   │
│  │  Statistical Parity Difference                     │   │
│  │  Disparate Impact                                   │   │
│  │  Mean Difference
      Shap
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  🤖 STEP 6: MULTI-LLM BIAS ANALYSIS                                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │   llama_3_3     │ │   deepseek_r1   │ │   mistral_nemo  │              │
│  │                 │ │                 │ │                 │              │
│  │ • Bias Level    │ │ • Bias Level    │ │ • Bias Level    │              │
│  │ • Definition    │ │ • Definition    │ │ • Definition    │              │
│  │ • Argumentation │ │ • Argumentation │ │ • Argumentation │              │
│  │ • Patterns      │ │ • Patterns      │ │ • Patterns      │              │
│  │ • Risk          │ │ • Risk          │ │ • Risk          │              │
│  │ • Features      │ │ • Features      │ │ • Features      │              │
│  │ • Proxy         │ │ • Proxy         │ │ • Proxy         │              │
│  │   Detection     │ │   Detection     │ │   Detection     │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  📋 STEP 7: RESULTS INTEGRATION & REPORTING                               │
│  • Web Interface: Interactive bias analysis                               │
│  • Standalone Reports: SHAP Global Explanations HTML                      │
│  • Multi-LLM Comparison: Consensus analysis                               │

└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 **Key Features**

### **🤖 Multi-LLM Analysis**
- **llama_3_3**: Meta's latest model for comprehensive analysis
- **deepseek_r1**: DeepSeek's reasoning-focused model
- **mistral_nemo**: Mistral's efficient analysis model
- **Consensus Building**: Compare insights across all three models
- **Key Findings**: Each LLM provides bias level, definition, and argumentation in 3 sentences max

### **📊 SHAP Global Explanations**
- **Per-Class Analysis**: Separate feature importance for each target class
- **SHAP Values**: True SHAP-based feature importance scores
- **Standalone Reports**: Beautiful HTML reports with detailed visualizations
- **Automatic Generation**: Reports created automatically when files are uploaded

### **⚖️ AIF360 Fairness Metrics**
- **Statistical Parity Difference**: Outcome rate differences between groups
- **Disparate Impact**: Ratio of positive outcomes between groups
- **Mean Difference**: Average outcome differences between groups
- **Bias Level Classification**: Scientific scoring system (0-9 points)

### **🎯 Automated Pipeline**
- **Column Detection**: LLMs automatically identify protected attributes
- **Target Identification**: Smart detection of target variables
- **Feature Selection**: Focus on demographic and protected attributes
- **Bias Classification**: Automated bias level assessment

## 📁 **Project Structure**

```
dsai2/
├── 📁 app/                          # Main Flask application
│   ├── 📁 models/
│   │   └── bias_evaluator.py        # Core bias analysis & AIF360 metrics
│   ├── 📁 routes/
│   │   └── main.py                  # Web routes & API endpoints
│   ├── 📁 utils/
│   │   └── llm_utils.py             # Multi-LLM integration & API calls
│   └── config.py                    # LLM endpoint configurations
├── 📁 templates/
│   └── index.html                   # Web interface with results display
├── 📁 uploads/                      # Uploaded dataset storage
├── 📁 reports/                      # Generated SHAP HTML reports
├── 🐍 generate_global_explanations_report.py  # SHAP report generator
├── 🐍 auto_generate_report.py       # Automatic report generation
├── 🐍 run.py                        # Flask app entry point
└── 📋 requirements.txt              # Python dependencies
```

## 🛠️ **Installation & Setup**

### **1. Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### **2. Start the Main Application**
   ```bash
   python run.py
   ```
Access at: `http://localhost:5000`

### **3. Start Automatic Report Generation (Optional)**
```bash
python auto_generate_report.py
```

## 📖 **Usage Guide**

### **Web Interface**
1. **Upload CSV file** through the web interface
2. **Select analysis mode**:
   - Single LLM: Choose one model for analysis
   - Multi-LLM: Compare all three models
3. **View results**:
   - Overall bias assessment
   - Per-attribute analysis
   - Multi-LLM insights
   - SHAP feature importance

### **Standalone Reports**
```bash
# Generate SHAP report for specific file
python generate_global_explanations_report.py your_dataset.csv

# With options
python generate_global_explanations_report.py your_dataset.csv --n-rows 1000
```

### **Automatic Monitoring**
```bash
# Start monitoring uploads directory
python auto_generate_report.py
# Reports generated automatically when files are uploaded
```

## 🔧 **Configuration**

### **LLM Endpoints** (`app/config.py`)
```python
LLM_ENDPOINTS = {
    'llama_3_3': {
        'url': 'https://api.groq.com/openai/v1/chat/completions',
        'headers': {'Authorization': 'Bearer YOUR_API_KEY'}
    },
    'deepseek_r1': {
        'url': 'https://api.deepseek.com/v1/chat/completions',
        'headers': {'Authorization': 'Bearer YOUR_API_KEY'}
    },
    'mistral_nemo': {
        'url': 'https://api.mistral.ai/v1/chat/completions',
        'headers': {'Authorization': 'Bearer YOUR_API_KEY'}
    }
}
```

### **Feature Selection** (in SHAP report generator)
```python
allowed_features = [
    'SUSP_RACE', 'SUSP_SEX', 'SUSP_AGE_GROUP',
    'VIC_RACE', 'VIC_SEX', 'VIC_AGE_GROUP'
]
```

## 📊 **Output Examples**

### **Bias Level Classification**
- 🟢 **LOW**: Minimal bias detected (0-2 points)
- 🟡 **MEDIUM**: Moderate bias detected (3-4 points)
- 🟠 **HIGH**: Significant bias detected (5-6 points)
- 🔴 **CRITICAL**: Severe bias detected (7-9 points)

### **SHAP Global Explanations**
- **Per-class feature importance** tables
- **SHAP value rankings** with percentages
- **Color-coded importance levels**
- **Standalone HTML reports**

### **Multi-LLM Insights**
- **Consensus analysis** across all three models
- **Model-specific bias interpretations**
- **Feature proxy detection**
- **Risk assessment** from each model
- **Comparison table** with AIF360 bias levels side-by-side

## 🔍 **Supported Datasets**

The system works with any CSV dataset containing:
- **Demographic columns** (race, sex, age, etc.)
- **Protected attributes** (automatically detected by LLMs)
- **Target variables** (crime categories, outcomes, etc.)

**Tested with:**
- NYPD Crime Data
- LAPD Crime Data
- Any law enforcement or demographic dataset

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different datasets
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 **Troubleshooting**

### **Common Issues**
- **LLM API Errors**: Check API keys and rate limits
- **Memory Issues**: Use `--n-rows` to limit dataset size
- **SHAP Calculation Errors**: System automatically falls back to feature importance
- **Report Generation**: Check file permissions in reports directory

### **Performance Tips**
- **Large Datasets**: Preprocess with `preprocess_dataset.py`
- **SHAP Speed**: Uses 500 samples by default for efficiency
- **Caching**: Uploaded files are cached for faster re-analysis

---

**🎯 Built for robust, automated bias detection using the latest AI technologies!**
1. Upload a CSV file through the web interface
2. The system will automatically:
   - Identify protected attributes
   - Determine target variables
   - Calculate fairness metrics
3. View the analysis results:
   - LLM recommendations
   - Protected attributes
   - Target variable
   - Fairness metrics by group

## Dependencies

- Flask: Web framework
- pandas: Data manipulation
- scikit-learn: Machine learning and metrics
- numpy: Numerical computations
- Bootstrap: Frontend styling

## Notes

- The application uses a factory pattern for better testing and configuration
- File caching is implemented to improve performance with large datasets
- Large datasets should be preprocessed using `preprocess_dataset.py` before analysis