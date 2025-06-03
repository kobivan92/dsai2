# Bias Analysis Tool

A Flask-based web application for analyzing potential biases in datasets, with a focus on fairness metrics across protected attributes.

## Project Structure

```
.
├── app/
│   ├── __init__.py           # Flask application factory
│   ├── models/
│   │   └── bias_evaluator.py # Core bias analysis logic
│   ├── routes/
│   │   └── main.py          # Route handlers
│   ├── static/
│   │   └── css/
│   │       └── style.css    # Custom styles
│   ├── templates/
│   │   ├── base.html        # Base template with common elements
│   │   └── index.html       # Main application page
│   └── utils/
│       └── llm_utils.py     # LLM integration utilities
├── uploads/                  # Directory for uploaded files
├── preprocess_dataset.py     # Dataset preprocessing script
├── requirements.txt          # Python dependencies
└── run.py                    # Application entry point
```

## Features

1. **Data Upload and Processing**
   - CSV file upload with automatic validation
   - File caching for improved performance
   - Dataset preprocessing capabilities

2. **Bias Analysis**
   - Protected attribute identification using LLM
   - Target variable determination
   - Correlation analysis
   - Fairness metrics calculation (precision, recall, F1-score)

3. **Results Visualization**
   - Overall performance metrics
   - Group-wise analysis
   - Interactive charts
   - Detailed metrics display

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python run.py
   ```

3. Access the application at `http://localhost:5000`

## Usage

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