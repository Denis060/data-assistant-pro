# Data Assistant Pro: Streamlit Data Analysis & AutoML App

A Streamlit application for data loading, quality assessment, cleaning, exploratory analysis, and machine learning.

## 📚 Documentation & Guides

**Quick Start**: 
- 📖 [Complete User Guide](USER_GUIDE.md) - Comprehensive step-by-step guide covering all features
- 🎯 [Interactive Walkthrough](WALKTHROUGH_GUIDE.md) - Video scripts, tutorials, and workshop materials  
- ⚡ [Quick Reference](QUICK_REFERENCE.md) - Cheat sheets, shortcuts, and troubleshooting guide

**New Users**: Start with the [User Guide](USER_GUIDE.md) for complete instructions or the [Quick Reference](QUICK_REFERENCE.md) for immediate answers.

## ✨ Key Features

### 🔍 **Smart Data Loading**
- **Intelligent CSV delimiter detection** - Automatically detects commas, semicolons, tabs, and pipes
- **Comprehensive file validation** - File size limits, format verification, encoding detection  
- **Error handling & logging** - Detailed logs for all data operations
- **Sample data included** - Ready-to-use housing dataset for testing

### 🧹 **Advanced Data Quality & Cleaning**
- **5-Dimensional Quality Assessment**:
  - **Completeness**: Missing value analysis with heatmaps
  - **Validity**: Data type consistency and format validation
  - **Consistency**: Pattern analysis and standardization checks
  - **Uniqueness**: Duplicate detection and ID validation
  - **Accuracy**: Statistical outlier detection and range validation

- **Domain-Specific Validation**:
  - **Demographics**: Age, gender, and population data validation
  - **Financial**: Salary, income, and monetary value checks
  - **Temporal**: Date consistency and logical relationship validation
  - **Measurements**: Height, weight, distance, and unit validation
  - **Business Logic**: ID uniqueness, categorical consistency, sum relationships

- **Smart Auto-Cleaning Pipeline**:
  - Intelligent missing value imputation (mean, median, mode, forward-fill)
  - Automated outlier detection and treatment (IQR, Z-score methods)
  - Duplicate removal with configurable strategies
  - Data type optimization for memory efficiency
  - One-click comprehensive cleaning

### 📊 **Comprehensive EDA (Exploratory Data Analysis)**
- **Automated Statistical Analysis**:
  - Descriptive statistics with confidence intervals
  - Distribution analysis and normality testing
  - Correlation matrices with significance testing
  - Feature importance analysis

- **Interactive Visualizations**:
  - Distribution plots (histograms, box plots, violin plots)
  - Correlation heatmaps with clustering
  - Missing value patterns visualization
  - Outlier detection plots
  - Feature relationship scatter plots

### 🤖 **Advanced Machine Learning Pipeline**
- **Multi-Algorithm Support**:
  - **Classification**: Random Forest, SVM, Logistic Regression, Gradient Boosting
  - **Regression**: Random Forest, SVR, Linear Regression, Ridge, Lasso

- **Professional ML Workflow**:
  - Automated feature engineering and preprocessing
  - Smart train/validation/test splitting
  - Hyperparameter optimization with cross-validation
  - Model performance comparison with statistical significance
  - Feature importance analysis
  - Prediction confidence intervals

- **Model Evaluation**:
  - **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
  - **Regression**: RMSE, MAE, R², Cross-validation scores
  - Learning curves and validation curves
  - Model interpretability reports

### 📈 **Interactive Predictions**
- **Single Prediction Interface**: Input values through dynamic forms
- **Batch Prediction System**: Upload CSV files for bulk predictions
- **Probability Estimates**: Classification confidence scores
- **Prediction Export**: Download results as CSV with timestamps

### 📁 **Data Export & Reporting**
- **Multiple Export Formats**: CSV, Excel, JSON
- **Comprehensive Reports**: Include cleaning steps, quality scores, and transformations
- **Prediction Results**: Downloadable prediction reports with metadata
- **Quality Audit Trail**: Complete log of all data quality improvements

## 🛠 **Engineering Features**

### ⚙️ **Configuration Management**
- Environment-based configuration (development/production)
- Configurable file size limits and processing parameters
- Logging level management
- ML model parameter tuning

### 🧪 **Testing Infrastructure**
- **Comprehensive Test Suite**: 17+ unit tests covering all modules
- **Quality Assurance**: Automated code formatting (Black, isort)
- **Linting**: Flake8, mypy for code quality
- **Security Scanning**: Bandit for vulnerability detection
- **Pre-commit Hooks**: Automated quality checks

### 🐳 **DevOps & Deployment**
- **Docker Containerization**: Docker and docker-compose setup
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment  
- **Development Tools**: Makefile for common tasks
- **Environment Management**: Virtual environment with pinned dependencies

### 📋 **Code Quality**
- **Type Hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings and inline comments
- **Error Handling**: Robust exception handling with user-friendly messages
- **Logging**: Structured logging for debugging and monitoring
- **Security**: Input validation and secure file handling

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd data-assistant-pro
```

2. **Set up virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the application**:
```bash
streamlit run app.py
```

5. **Open in browser**: Navigate to `http://localhost:8501`

### Using Docker

1. **Build and run with Docker Compose**:
```bash
docker-compose up --build
```

2. **Access the application**: `http://localhost:8501`

## 📖 **Usage Guide**

### 1. **Data Loading**
- Upload CSV files or use the sample housing dataset
- Review data quality metrics and validation results
- Examine the automatically generated data profile

### 2. **Data Quality Assessment**
- Navigate to the "🧹 Data Cleaning" tab
- Review the comprehensive quality dashboard
- Check domain-specific validation results
- Identify data quality issues and recommendations

### 3. **Data Cleaning**
- Use the "Auto-Clean" features for one-click cleaning
- Or manually select specific cleaning operations
- Monitor cleaning impact with before/after comparisons
- Export cleaned data for further analysis

### 4. **Exploratory Data Analysis**
- Automatic generation of comprehensive EDA reports
- Interactive visualizations and statistical summaries
- Correlation analysis and feature relationships
- Missing value and outlier analysis

### 5. **Machine Learning**
- Select target variable and features
- Choose between classification and regression
- Compare multiple algorithms automatically
- Evaluate model performance with detailed metrics
- Make predictions on new data

### 6. **Export & Reporting**
- Download cleaned datasets in multiple formats
- Export prediction results with confidence scores
- Generate comprehensive analysis reports

## 🏗 **Architecture**

```
data-assistant-pro/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Multi-container setup
├── Makefile                 # Development tasks
├── pyproject.toml           # Project configuration
├── .pre-commit-config.yaml  # Pre-commit hooks
├── .github/workflows/       # CI/CD pipeline
├── data/                    # Sample datasets
├── modules/                 # Core functionality
│   ├── loader.py           # Smart data loading
│   ├── data_quality.py     # Quality assessment
│   ├── domain_validation.py # Business logic validation
│   ├── cleaning_fixed.py   # Data cleaning operations
│   ├── eda.py              # Exploratory data analysis
│   └── modeling.py         # Machine learning pipeline
└── tests/                   # Test suite
    ├── test_loader.py
    ├── test_cleaning.py
    ├── test_eda.py
    └── test_modeling.py
```

## 🧪 **Development**

### Running Tests
```bash
# Run all tests
make test

# Run with coverage
pytest --cov=modules tests/

# Run specific test file
pytest tests/test_loader.py
```

### Code Quality
```bash
# Format code
make format

# Run linting
make lint

# Security scan
make security
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## 📊 **Performance & Scalability**

- **Memory Optimization**: Efficient data type conversion and chunked processing
- **Large File Support**: Configurable memory limits and streaming processing
- **Caching**: Streamlit caching for expensive operations
- **Background Processing**: Non-blocking operations for large datasets
- **Resource Monitoring**: Memory and CPU usage tracking

## 🔒 **Security Features**

- **Input Validation**: Comprehensive file and data validation
- **File Size Limits**: Configurable upload size restrictions  
- **Secure File Handling**: Safe file processing and cleanup
- **Error Sanitization**: No sensitive information in error messages
- **Dependency Security**: Regular security scanning with Bandit

## 📝 **Logging & Monitoring**

- **Structured Logging**: JSON-formatted logs for easy parsing
- **Error Tracking**: Detailed error logs with context
- **Performance Metrics**: Processing time and resource usage
- **Audit Trail**: Complete record of data transformations
- **Debug Mode**: Verbose logging for troubleshooting

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks (`make test lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Streamlit** for the amazing web app framework
- **Pandas** for powerful data manipulation
- **Scikit-learn** for machine learning capabilities
- **Plotly** for interactive visualizations
- **Docker** for containerization support

---

## 🔮 **Future Enhancements**

- **Database Integration**: PostgreSQL, MySQL, MongoDB support
- **Time Series Analysis**: Advanced temporal data analysis
- **Deep Learning**: Neural network integration with TensorFlow/PyTorch
- **Real-time Data**: Streaming data processing capabilities
- **API Integration**: RESTful API for programmatic access
- **Advanced Visualizations**: 3D plots, geospatial analysis
- **Automated Feature Engineering**: Advanced feature creation
- **Model Deployment**: One-click model deployment to cloud platforms

**Built with ❤️ for data professionals who demand excellence.**
