"""
REST API Integration for Data Assistant Pro
Expose core functionality through REST endpoints
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import io
import json
import logging
from datetime import datetime
import uuid
import os
import asyncio

# Import our modules
import sys
sys.path.append('.')
from modules.data_quality import DataQualityChecker, SmartDataCleaner
from modules.domain_validation import DataValidationRules
from modules.time_series import TimeSeriesAnalyzer

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Assistant Pro API",
    description="Enterprise-grade data analysis, cleaning, and machine learning API",
    version="1.0.0"
)

# Pydantic models for API requests/responses
class DataQualityRequest(BaseModel):
    data: Dict[str, List[Any]]
    
class DataQualityResponse(BaseModel):
    overall_score: float
    completeness: Dict[str, Any]
    validity: Dict[str, Any]
    consistency: Dict[str, Any]
    uniqueness: Dict[str, Any]
    accuracy: Dict[str, Any]
    issues: List[str]
    recommendations: List[str]

class CleaningRequest(BaseModel):
    data: Dict[str, List[Any]]
    options: Optional[Dict[str, Any]] = {}

class ValidationRequest(BaseModel):
    data: Dict[str, List[Any]]

class TimeSeriesRequest(BaseModel):
    data: Dict[str, List[Any]]
    date_column: str
    value_column: str
    forecast_steps: Optional[int] = 30

class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    model_name: Optional[str] = None

# Global storage for sessions and models
sessions = {}
trained_models = {}

@app.get("/")
async def root():
    """API health check and information."""
    return {
        "message": "Data Assistant Pro API",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": {
            "/data-quality": "Assess data quality",
            "/clean-data": "Clean and process data",
            "/validate-data": "Domain-specific validation",
            "/time-series": "Time series analysis and forecasting",
            "/upload": "Upload data files",
            "/sessions": "Manage data sessions"
        }
    }

@app.post("/data-quality", response_model=DataQualityResponse)
async def assess_data_quality(request: DataQualityRequest):
    """Assess data quality using 5-dimensional analysis."""
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Perform quality assessment
        checker = DataQualityChecker(df)
        quality_report = checker.assess_overall_quality()
        
        return DataQualityResponse(**quality_report)
        
    except Exception as e:
        logger.error(f"Error in data quality assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")

@app.post("/clean-data")
async def clean_data(request: CleaningRequest):
    """Clean data using smart auto-cleaning pipeline."""
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Perform auto-cleaning
        cleaner = SmartDataCleaner(df)
        cleaned_df, cleaning_report = cleaner.auto_clean_pipeline()
        
        return {
            "cleaned_data": cleaned_df.to_dict(orient="records"),
            "cleaning_report": cleaning_report,
            "original_shape": df.shape,
            "cleaned_shape": cleaned_df.shape,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data cleaning failed: {str(e)}")

@app.post("/validate-data")
async def validate_data(request: ValidationRequest):
    """Perform domain-specific data validation."""
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Perform validation
        validator = DataValidationRules(df)
        validation_results = validator.validate_all()
        
        return {
            "validation_results": validation_results,
            "total_issues": validation_results['total_issues'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in data validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data validation failed: {str(e)}")

@app.post("/time-series")
async def analyze_time_series(request: TimeSeriesRequest):
    """Perform time series analysis and forecasting."""
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Perform time series analysis
        analyzer = TimeSeriesAnalyzer(df)
        
        # Prepare time series
        ts_data = analyzer.prepare_time_series(
            request.date_column, 
            request.value_column, 
            'D'
        )
        
        # Generate forecast
        forecast_result = analyzer.forecast_arima(
            ts_data, 
            order=(1, 1, 1), 
            steps=request.forecast_steps
        )
        
        # Detect anomalies
        anomalies = analyzer.detect_anomalies(ts_data, 'iqr')
        
        return {
            "time_series_data": ts_data.to_dict(),
            "forecast": forecast_result.get('forecast', {}).to_dict() if 'forecast' in forecast_result else {},
            "anomalies": anomalies.to_dict(),
            "anomaly_count": anomalies.sum(),
            "data_points": len(ts_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in time series analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Time series analysis failed: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process data files."""
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Create session ID
        session_id = str(uuid.uuid4())
        
        # Process file based on type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = json.loads(contents.decode('utf-8'))
            df = pd.DataFrame(data)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Store in session
        sessions[session_id] = {
            'data': df,
            'filename': file.filename,
            'upload_time': datetime.now(),
            'shape': df.shape
        }
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "sample_data": df.head().to_dict(orient="records")
        }
        
    except Exception as e:
        logger.error(f"Error in file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    df = session['data']
    
    return {
        "session_id": session_id,
        "filename": session['filename'],
        "upload_time": session['upload_time'].isoformat(),
        "shape": session['shape'],
        "columns": list(df.columns),
        "data_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_data": df.head().to_dict(orient="records")
    }

@app.post("/sessions/{session_id}/quality")
async def session_quality_assessment(session_id: str):
    """Assess quality for session data."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = sessions[session_id]['data']
    
    # Perform quality assessment
    checker = DataQualityChecker(df)
    quality_report = checker.assess_overall_quality()
    
    return quality_report

@app.post("/sessions/{session_id}/clean")
async def session_clean_data(session_id: str):
    """Clean data for session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = sessions[session_id]['data']
    
    # Perform auto-cleaning
    cleaner = SmartDataCleaner(df)
    cleaned_df, cleaning_report = cleaner.auto_clean_pipeline()
    
    # Update session with cleaned data
    sessions[session_id]['cleaned_data'] = cleaned_df
    sessions[session_id]['cleaning_report'] = cleaning_report
    
    return {
        "session_id": session_id,
        "cleaning_report": cleaning_report,
        "original_shape": df.shape,
        "cleaned_shape": cleaned_df.shape
    }

@app.get("/sessions/{session_id}/export")
async def export_session_data(session_id: str, format: str = "csv"):
    """Export session data."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get cleaned data if available, otherwise original
    session = sessions[session_id]
    df = session.get('cleaned_data', session['data'])
    
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data_export_{session_id[:8]}_{timestamp}.{format}"
    
    if format == "csv":
        # Save as CSV
        df.to_csv(filename, index=False)
        return FileResponse(
            filename, 
            media_type='text/csv',
            filename=filename
        )
    elif format == "json":
        # Save as JSON
        with open(filename, 'w') as f:
            df.to_json(f, orient='records', indent=2)
        return FileResponse(
            filename,
            media_type='application/json',
            filename=filename
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted successfully"}

@app.get("/health")
async def health_check():
    """API health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(sessions),
        "trained_models": len(trained_models)
    }

# Background task for cleanup
async def cleanup_old_sessions():
    """Clean up old sessions periodically."""
    while True:
        current_time = datetime.now()
        to_delete = []
        
        for session_id, session in sessions.items():
            if (current_time - session['upload_time']).hours > 24:  # 24 hour expiry
                to_delete.append(session_id)
        
        for session_id in to_delete:
            del sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
        
        await asyncio.sleep(3600)  # Check every hour

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks."""
    # Start cleanup task
    asyncio.create_task(cleanup_old_sessions())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
