# Data Assistant Pro API Documentation

## Overview
The Data Assistant Pro API provides programmatic access to enterprise-grade data analysis, cleaning, and machine learning capabilities. Built with FastAPI, it offers high-performance REST endpoints for integrating data science workflows into your applications.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, the API does not require authentication. In production environments, consider implementing API keys or OAuth2.

## API Endpoints

### Health Check
```http
GET /
GET /health
```
Check API status and get basic information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "active_sessions": 5,
  "trained_models": 3
}
```

### Data Quality Assessment
```http
POST /data-quality
```
Perform comprehensive 5-dimensional data quality analysis.

**Request Body:**
```json
{
  "data": {
    "column1": [1, 2, 3, null, 5],
    "column2": ["A", "B", "C", "D", "E"]
  }
}
```

**Response:**
```json
{
  "overall_score": 85.5,
  "completeness": {
    "score": 90.0,
    "missing_percentage": 10.0
  },
  "validity": {
    "score": 95.0,
    "invalid_values": 2
  },
  "consistency": {
    "score": 88.0,
    "inconsistencies": 3
  },
  "uniqueness": {
    "score": 92.0,
    "duplicate_percentage": 8.0
  },
  "accuracy": {
    "score": 87.0,
    "accuracy_issues": 5
  },
  "issues": ["Missing values in column1", "Inconsistent format in column2"],
  "recommendations": ["Handle missing values", "Standardize text format"]
}
```

### Data Cleaning
```http
POST /clean-data
```
Clean data using intelligent auto-cleaning pipeline.

**Request Body:**
```json
{
  "data": {
    "column1": [1, 2, 3, null, 5],
    "column2": ["A", "B", "C", "D", "E"]
  },
  "options": {
    "handle_missing": true,
    "remove_outliers": true
  }
}
```

**Response:**
```json
{
  "cleaned_data": [
    {"column1": 1, "column2": "A"},
    {"column1": 2, "column2": "B"},
    {"column1": 3, "column2": "C"},
    {"column1": 4, "column2": "D"},
    {"column1": 5, "column2": "E"}
  ],
  "cleaning_report": {
    "operations_applied": ["missing_value_imputation", "outlier_removal"],
    "rows_affected": 1,
    "columns_affected": 1
  },
  "original_shape": [5, 2],
  "cleaned_shape": [5, 2],
  "timestamp": "2024-01-15T10:30:00"
}
```

### Data Validation
```http
POST /validate-data
```
Perform domain-specific data validation.

**Request Body:**
```json
{
  "data": {
    "email": ["user@example.com", "invalid-email"],
    "age": [25, 150, -5]
  }
}
```

**Response:**
```json
{
  "validation_results": {
    "email_validation": {
      "valid_count": 1,
      "invalid_count": 1,
      "issues": ["Invalid email format: invalid-email"]
    },
    "age_validation": {
      "valid_count": 1,
      "invalid_count": 2,
      "issues": ["Age out of range: 150", "Negative age: -5"]
    },
    "total_issues": 3
  },
  "total_issues": 3,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Time Series Analysis
```http
POST /time-series
```
Analyze time series data with forecasting and anomaly detection.

**Request Body:**
```json
{
  "data": {
    "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "value": [100, 105, 98]
  },
  "date_column": "date",
  "value_column": "value",
  "forecast_steps": 30
}
```

**Response:**
```json
{
  "time_series_data": {
    "2024-01-01": 100,
    "2024-01-02": 105,
    "2024-01-03": 98
  },
  "forecast": {
    "2024-01-04": 102.5,
    "2024-01-05": 103.2
  },
  "anomalies": {
    "2024-01-01": false,
    "2024-01-02": false,
    "2024-01-03": true
  },
  "anomaly_count": 1,
  "data_points": 3,
  "timestamp": "2024-01-15T10:30:00"
}
```

## File Upload and Session Management

### Upload File
```http
POST /upload
```
Upload a data file and create a new session.

**Form Data:**
- `file`: CSV or JSON file

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "data.csv",
  "shape": [1000, 10],
  "columns": ["col1", "col2", "col3"],
  "data_types": {"col1": "int64", "col2": "object"},
  "sample_data": [
    {"col1": 1, "col2": "A"},
    {"col1": 2, "col2": "B"}
  ]
}
```

### Get Session Information
```http
GET /sessions/{session_id}
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "data.csv",
  "upload_time": "2024-01-15T10:30:00",
  "shape": [1000, 10],
  "columns": ["col1", "col2"],
  "data_types": {"col1": "int64", "col2": "object"},
  "missing_values": {"col1": 0, "col2": 5},
  "sample_data": [...]
}
```

### Session-based Operations
```http
POST /sessions/{session_id}/quality
POST /sessions/{session_id}/clean
GET /sessions/{session_id}/export?format=csv
DELETE /sessions/{session_id}
```

## Python Client Library

Install the client:
```python
from modules.api_client import DataAssistantAPI

# Initialize client
api = DataAssistantAPI("http://localhost:8000")

# Check health
status = api.health_check()

# Assess data quality
import pandas as pd
df = pd.read_csv("your_data.csv")
quality_report = api.assess_data_quality(df)

# Clean data
cleaning_result = api.clean_data(df)
cleaned_df = cleaning_result['cleaned_data']

# Time series analysis
ts_result = api.analyze_time_series(
    df, 
    date_column="date", 
    value_column="value",
    forecast_steps=30
)
```

## Error Handling

The API uses standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found (session not found)
- `422` - Validation Error
- `500` - Internal Server Error

Error Response Format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting
Currently no rate limiting is implemented. For production use, consider implementing rate limiting based on your requirements.

## Performance Considerations

- **File Size**: Recommended maximum file size is 100MB
- **Memory Usage**: Large datasets may require significant memory
- **Session Cleanup**: Sessions expire after 24 hours of inactivity
- **Concurrent Requests**: API supports multiple concurrent requests

## Examples

### Complete Workflow Example
```python
import pandas as pd
from modules.api_client import DataAssistantAPI

# Initialize API client
api = DataAssistantAPI()

# Upload file and create session
session_info = api.upload_file("data.csv")
session_id = session_info["session_id"]

# Assess data quality
quality_report = api.session_quality_assessment(session_id)
print(f"Data Quality Score: {quality_report['overall_score']}")

# Clean the data
cleaning_result = api.session_clean_data(session_id)
print(f"Cleaning completed: {cleaning_result['cleaning_report']}")

# Export cleaned data
cleaned_data = api.export_session_data(session_id, format="csv")
with open("cleaned_data.csv", "wb") as f:
    f.write(cleaned_data)

# Clean up session
api.delete_session(session_id)
```

### Batch Processing Example
```python
import os
from pathlib import Path

# Process multiple files
data_dir = Path("data_files")
results = []

for file_path in data_dir.glob("*.csv"):
    # Upload and process each file
    session_info = api.upload_file(str(file_path))
    session_id = session_info["session_id"]
    
    # Get quality assessment
    quality = api.session_quality_assessment(session_id)
    
    # Store results
    results.append({
        "filename": file_path.name,
        "quality_score": quality["overall_score"],
        "shape": session_info["shape"]
    })
    
    # Clean up
    api.delete_session(session_id)

# Summary report
for result in results:
    print(f"{result['filename']}: Quality {result['quality_score']:.1f}%")
```

## Interactive API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation with Swagger UI, where you can test endpoints directly from your browser.

## Support
For support and questions, please refer to the main project documentation or create an issue in the project repository.
