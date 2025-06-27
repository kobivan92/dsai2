# Global Explanations Report Generator

This tool automatically generates beautiful, standalone HTML reports with Global Explanations whenever you upload a new CSV file.

## Features

- 🎯 **Automatic Detection**: Monitors the uploads directory for new CSV files
- 📊 **Global Explanations**: Shows feature importance for each class
- 🎨 **Beautiful Reports**: Professional HTML reports with Bootstrap styling
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🔄 **Real-time Generation**: Creates reports automatically when files are uploaded

## Quick Start

### Option 1: Automatic Monitoring (Recommended)

1. **Start the monitoring service**:
   ```bash
   python auto_generate_report.py
   ```

2. **Upload a CSV file** to the `uploads` directory (or use the main app)

3. **Report is automatically generated** and opened in your browser

### Option 2: Manual Generation

Generate a report for a specific file:

```bash
python generate_global_explanations_report.py your_file.csv
```

**Options**:
- `--n-rows 1000`: Limit analysis to first 1000 rows
- `--test-size 0.2`: Set test size (default: 0.2)
- `--max-categories 50`: Set max categories (default: 50)
- `--output-dir reports`: Set output directory

## Report Features

### 📋 Dataset Information
- File details (name, rows, columns)
- Target column identification
- Protected attributes analysis

### 📊 Global Explanations
- **Per-class analysis**: Separate tables for each class
- **Feature ranking**: Top 10 most important features per class
- **Importance scores**: Numerical and percentage values
- **Color coding**: 
  - 🟢 High importance (>20%)
  - 🟡 Medium importance (10-20%)
  - ⚪ Low importance (<10%)

### 🎨 Visual Design
- Professional Bootstrap styling
- Responsive tables
- Color-coded importance levels
- Icons and badges for better UX

## File Structure

```
reports/
├── global_explanations_report_20241201_143022.html
├── global_explanations_report_20241201_143156.html
└── ...

uploads/
├── your_dataset.csv
└── another_dataset.csv
```

## Example Usage

1. **Start monitoring**:
   ```bash
   python auto_generate_report.py
   ```

2. **Upload a file** through the main app or copy to uploads directory

3. **Report is automatically generated** with timestamp:
   ```
   ✅ Report generated: reports/global_explanations_report_20241201_143022.html
   🌐 Report opened in browser
   ```

## Report Content

Each report includes:

1. **Header Section**
   - Report title and description
   - Generation timestamp

2. **Dataset Summary**
   - File information
   - Row and column counts
   - Target column
   - Protected attributes

3. **Global Explanations Tables**
   - One table per class
   - Feature rankings
   - Importance scores
   - Percentage values

4. **Legend**
   - Explanation of importance levels
   - Understanding guide

## Troubleshooting

### Report not generating?
- Check if the file is a valid CSV
- Ensure the file is completely uploaded
- Check console for error messages

### Report looks empty?
- The dataset might not have enough data
- Check if target column was identified correctly
- Verify protected attributes were found

### Performance issues?
- Use `--n-rows` to limit dataset size
- Reduce `--max-categories` for large datasets
- Consider using smaller test size

## Integration with Main App

The report generator works alongside the main Flask app:

1. **Main app**: Upload files and get bias analysis
2. **Report generator**: Automatically creates standalone reports
3. **Both**: Use the same uploads directory

This gives you both the interactive web interface and beautiful offline reports! 