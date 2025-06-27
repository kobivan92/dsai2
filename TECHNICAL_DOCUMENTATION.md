# 🔧 Technical Documentation: Multi-LLM Bias Detection System

Comprehensive technical documentation for the bias detection and analysis pipeline, including architecture details, API specifications, and implementation guides.

## 🏗️ **System Architecture**

### **High-Level Architecture**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM COMPONENTS                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Report Generator│    │  Auto Monitor   │
│   (Flask App)   │    │  (Standalone)   │    │  (File Watch)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Core Pipeline  │
                    │                 │
                    │ • LLM Analysis  │
                    │ • SHAP Calc     │
                    │ • Bias Metrics  │
                    │ • Classification│
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Data Storage   │
                    │                 │
                    │ • uploads/      │
                    │ • reports/      │
                    │ • file_cache/   │
                    └─────────────────┘
```

### **Component Details**

#### **1. Web Interface (Flask App)**
- **Framework**: Flask 2.3.3
- **Templates**: Bootstrap 5.1.3 + Font Awesome
- **Routes**: `/` (home), `/analyze` (single LLM), `/analyze_multi` (multi-LLM)
- **File Handling**: Upload, cache, and process CSV files

#### **2. Report Generator (Standalone)**
- **SHAP Integration**: TreeExplainer with Random Forest
- **HTML Generation**: Bootstrap-styled standalone reports
- **Feature Focus**: Demographic attributes only
- **Error Handling**: Graceful fallback to feature importance

#### **3. Auto Monitor (File Watch)**
- **Library**: watchdog 3.0.0
- **Monitoring**: uploads/ directory
- **Threading**: Non-blocking report generation
- **Browser Integration**: Auto-open generated reports

## 🔍 **LLM Integration Architecture**

### **LLM Endpoints Configuration**
```python
LLM_ENDPOINTS = {
    'llama_3_3': {
        'url': 'https://api.groq.com/openai/v1/chat/completions',
        'headers': {
            'Authorization': 'Bearer YOUR_API_KEY',
            'Content-Type': 'application/json'
        },
        'model': 'llama-3.1-8b-instant'
    },
    'deepseek_r1': {
        'url': 'https://api.deepseek.com/v1/chat/completions',
        'headers': {
            'Authorization': 'Bearer YOUR_API_KEY',
            'Content-Type': 'application/json'
        },
        'model': 'deepseek-chat'
    },
    'mistral_nemo': {
        'url': 'https://api.mistral.ai/v1/chat/completions',
        'headers': {
            'Authorization': 'Bearer YOUR_API_KEY',
            'Content-Type': 'application/json'
        },
        'model': 'mistral-large-latest'
    }
}
```

### **LLM Analysis Pipeline**

#### **Step 1: Column Analysis**
```python
def get_llm_recommendations(columns_description, df, llm_model='llama_3_3'):
    """
    LLM analyzes dataset columns to identify:
    - Target column (what to predict)
    - Protected attributes (demographic features)
    - Excluded columns (irrelevant features)
    - Race column (for bias analysis)
    - Privileged/unprivileged groups
    """
```

#### **Step 2: Bias Analysis**
```python
def get_llm_bias_check(protected_attr, analysis_summary, global_explanations=None, llm_model='llama_3_3'):
    """
    LLM analyzes bias patterns using:
    - Performance metrics (precision, recall, F1)
    - Group-wise analysis
    - SHAP feature importance
    - Statistical bias metrics
    """
```

### **Multi-LLM Consensus Building**
```python
def get_llm_bias_check_multi(protected_attr, analysis_summary, global_explanations=None):
    """
    Runs bias analysis across all three LLMs:
    1. llama_3_3: Comprehensive analysis
    2. deepseek_r1: Reasoning-focused analysis
    3. mistral_nemo: Efficient analysis
    
    Returns consensus insights and model-specific interpretations
    """
```

## 📊 **SHAP Integration Details**

### **SHAP Calculation Pipeline**
```python
def calculate_shap_global_explanations(df, target_column, protected_attributes):
    """
    1. Data Preprocessing
       - Select demographic features only
       - Encode categorical variables
       - Handle missing values
    
    2. Model Training
       - Random Forest (100 estimators)
       - Multi-class classification
    
    3. SHAP Calculation
       - TreeExplainer for tree-based models
       - Sample 500 instances for efficiency
       - Calculate per-class SHAP values
    
    4. Feature Importance
       - Mean absolute SHAP values
       - Rank features by importance
       - Calculate percentages
    """
```

### **SHAP Array Handling**
```python
# Handle different SHAP output formats
if isinstance(shap_values, list):
    # Multi-class: list of arrays
    for class_idx, class_shap_values in enumerate(shap_values):
        mean_shap_values = np.abs(class_shap_values).mean(axis=0)
elif len(shap_values.shape) == 3:
    # 3D array: (samples, features, classes) or (classes, samples, features)
    if shap_values.shape[0] == sample_size:
        # Transpose to (classes, samples, features)
        shap_values = np.transpose(shap_values, (2, 0, 1))
    mean_shap_values = np.abs(shap_values).mean(axis=1)
```

## ⚖️ **AIF360 Fairness Metrics**

### **Bias Metrics Calculation**
```python
def compute_bias_metrics(df, target_col, race_col, privileged_list, unprivileged_list):
    """
    For each target category, calculate:
    
    1. Statistical Parity Difference (SPD)
       - Difference in positive outcome rates between groups
       - Fair: |SPD| < 0.05
       - Biased: |SPD| ≥ 0.05
    
    2. Disparate Impact (DI)
       - Ratio of positive outcome rates between groups
       - Fair: 0.8 < DI < 1.25
       - Biased: DI ≤ 0.8 or DI ≥ 1.25
    
    3. Mean Difference (MD)
       - Average difference in outcomes between groups
       - Fair: |MD| < 0.05
       - Biased: |MD| ≥ 0.05
    """
```

### **Bias Level Classification Algorithm**
```python
def classify_bias_level(statistical_parity_diff, disparate_impact, mean_diff):
    """
    Scoring System (0-9 points total):
    
    Statistical Parity Difference:
    - |SPD| < 0.05: 0 points
    - |SPD| < 0.1:  1 point
    - |SPD| < 0.2:  2 points
    - |SPD| ≥ 0.2:  3 points
    
    Disparate Impact:
    - DI ratio > 0.8: 0 points
    - DI ratio > 0.6: 1 point
    - DI ratio > 0.4: 2 points
    - DI ratio ≤ 0.4: 3 points
    
    Mean Difference:
    - |MD| < 0.05: 0 points
    - |MD| < 0.1:  1 point
    - |MD| < 0.2:  2 points
    - |MD| ≥ 0.2:  3 points
    
    Final Classification:
    - 0-2 points: LOW
    - 3-4 points: MEDIUM
    - 5-6 points: HIGH
    - 7-9 points: CRITICAL
    """
```

## 🔄 **Data Flow Pipeline**

### **Complete Pipeline Flow**
```
1. CSV Upload
   ↓
2. LLM Column Analysis (3 models)
   ↓
3. Data Preprocessing
   - Drop excluded columns
   - Handle missing values
   - Encode categorical features
   ↓
4. Model Training
   - Random Forest classifier
   - Multi-class classification
   ↓
5. SHAP Calculation
   - TreeExplainer
   - Per-class feature importance
   ↓
6. AIF360 Bias Metrics
   - Statistical Parity Difference
   - Disparate Impact
   - Mean Difference
   ↓
7. Bias Level Classification
   - Scoring system (0-9 points)
   - LOW/MEDIUM/HIGH/CRITICAL
   ↓
8. Multi-LLM Bias Analysis
   - Consensus building
   - Model-specific insights
   ↓
9. Results Integration
   - Web interface display
   - Standalone HTML reports
```

### **Error Handling Strategy**
```python
# Multi-level error handling
try:
    # Primary: SHAP calculation
    shap_values = explainer.shap_values(X_sample)
except Exception as e1:
    try:
        # Secondary: Expected values approach
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
    except Exception as e2:
        # Tertiary: Feature importance fallback
        return calculate_feature_importance_fallback(X, y, rf_model)
```

## 📁 **File Structure & Dependencies**

### **Core Dependencies**
```txt
flask==2.3.3              # Web framework
pandas==2.0.3             # Data manipulation
numpy==1.24.3             # Numerical computations
scikit-learn==1.3.0       # Machine learning
requests==2.31.0          # HTTP requests
aif360==0.5.0             # Fairness metrics
shap==0.43.0              # SHAP explanations
watchdog==3.0.0           # File monitoring
matplotlib==3.7.2         # Plotting
seaborn==0.12.2           # Statistical visualization
```

### **File Organization**
```
dsai2/
├── app/                          # Main application
│   ├── __init__.py              # Flask app factory
│   ├── config.py                # LLM endpoints & configuration
│   ├── models/
│   │   └── bias_evaluator.py    # Core bias analysis logic
│   ├── routes/
│   │   └── main.py              # Web routes & API endpoints
│   └── utils/
│       └── llm_utils.py         # LLM integration utilities
├── templates/
│   └── index.html               # Web interface
├── uploads/                     # Dataset storage
├── reports/                     # Generated HTML reports
├── file_cache/                  # Cached processed data
├── generate_global_explanations_report.py  # SHAP report generator
├── auto_generate_report.py      # Automatic report generation
└── run.py                       # Application entry point
```

## 🔧 **Configuration Management**

### **Environment Variables**
```bash
# LLM API Keys
GROQ_API_KEY=your_groq_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
MISTRAL_API_KEY=your_mistral_api_key

# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

### **Feature Configuration**
```python
# Demographic features for SHAP analysis
ALLOWED_FEATURES = [
    'SUSP_RACE', 'SUSP_SEX', 'SUSP_AGE_GROUP',
    'VIC_RACE', 'VIC_SEX', 'VIC_AGE_GROUP'
]

# SHAP calculation settings
SHAP_SAMPLE_SIZE = 500
RANDOM_FOREST_ESTIMATORS = 100
MAX_CATEGORIES = 50
```

## 🚀 **Performance Optimization**

### **Memory Management**
```python
# Limit dataset size for large files
if n_rows > 0:
    df = df.head(n_rows)

# Use sampling for SHAP calculation
sample_size = min(500, len(X))
X_sample = X.sample(n=sample_size, random_state=42)
```

### **Caching Strategy**
```python
# File caching for uploaded datasets
if os.path.exists(filepath):
    is_cached = True
else:
    file.save(filepath)
    is_cached = False
```

### **Parallel Processing**
```python
# Multi-threading for report generation
thread = threading.Thread(target=self.generate_report, args=(filepath,))
thread.daemon = True
thread.start()
```

## 🔒 **Security Considerations**

### **File Upload Security**
```python
# Validate file types
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

### **API Key Management**
```python
# Secure API key storage
import os
api_key = os.environ.get('GROQ_API_KEY')
if not api_key:
    raise ValueError("API key not found in environment variables")
```

## 📊 **Monitoring & Logging**

### **Logging Configuration**
```python
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('werkzeug')
log.setLevel(logging.INFO)
```

### **Performance Monitoring**
```python
# Track processing times
import time
start_time = time.time()
# ... processing ...
end_time = time.time()
print(f"Processing time: {end_time - start_time:.2f} seconds")
```

## 🧪 **Testing Strategy**

### **Unit Tests**
```python
def test_bias_level_classification():
    # Test bias level classification with known values
    assert classify_bias_level(0.01, 0.9, 0.02) == 'LOW'
    assert classify_bias_level(0.15, 0.3, 0.25) == 'CRITICAL'
```

### **Integration Tests**
```python
def test_end_to_end_pipeline():
    # Test complete pipeline with sample data
    # Upload file → LLM analysis → SHAP calculation → Report generation
```

## 🔄 **Deployment Guide**

### **Local Development**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=your_key
export DEEPSEEK_API_KEY=your_key
export MISTRAL_API_KEY=your_key

# Run application
python run.py
```

### **Production Deployment**
```bash
# Use production WSGI server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run:app

# Set up reverse proxy (nginx)
# Configure SSL certificates
# Set up monitoring and logging
```

## 📈 **Scalability Considerations**

### **Horizontal Scaling**
- **Load Balancing**: Multiple Flask instances
- **Database**: Add database for user management
- **Caching**: Redis for session and data caching
- **File Storage**: Cloud storage (S3, GCS)

### **Performance Optimization**
- **Async Processing**: Celery for background tasks
- **Batch Processing**: Process multiple files simultaneously
- **CDN**: Serve static assets from CDN
- **Database Indexing**: Optimize database queries

---

**🔧 This technical documentation provides comprehensive details for developers, system administrators, and researchers working with the bias detection system.** 
 