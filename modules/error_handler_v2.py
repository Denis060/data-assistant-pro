"""
Standardized Error Handling System
Provides consistent error handling, logging, and user feedback across all modules
"""

import logging
import traceback
import streamlit as st
from typing import Optional, Any, Dict, List
from enum import Enum
from datetime import datetime
import functools

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels for consistent handling."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better organization."""
    DATA_LOADING = "data_loading"
    DATA_PROCESSING = "data_processing"
    VALIDATION = "validation"
    MODELING = "modeling"
    VISUALIZATION = "visualization"
    SYSTEM = "system"
    USER_INPUT = "user_input"

class StandardizedError:
    """Standardized error representation."""
    
    def __init__(
        self, 
        message: str, 
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        technical_details: Optional[str] = None,
        suggested_action: Optional[str] = None,
        error_code: Optional[str] = None
    ):
        self.message = message
        self.severity = severity
        self.category = category
        self.technical_details = technical_details
        self.suggested_action = suggested_action
        self.error_code = error_code
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'technical_details': self.technical_details,
            'suggested_action': self.suggested_action,
            'error_code': self.error_code,
            'timestamp': self.timestamp.isoformat()
        }

class ErrorHandler:
    """Centralized error handling with consistent user feedback."""
    
    def __init__(self):
        self.error_history = []
        self.error_counts = {severity: 0 for severity in ErrorSeverity}
    
    def handle_error(
        self,
        error: Exception,
        message: str = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        suggested_action: str = None,
        show_technical_details: bool = False
    ) -> StandardizedError:
        """Handle an exception with standardized processing."""
        
        # Create standardized error
        user_message = message or f"An error occurred: {str(error)}"
        technical_details = f"{type(error).__name__}: {str(error)}"
        
        if show_technical_details:
            technical_details += f"\\n\\nTraceback:\\n{traceback.format_exc()}"
        
        std_error = StandardizedError(
            message=user_message,
            severity=severity,
            category=category,
            technical_details=technical_details,
            suggested_action=suggested_action
        )
        
        # Log the error
        self._log_error(std_error)
        
        # Show user feedback
        self._show_user_feedback(std_error)
        
        # Track error
        self.error_history.append(std_error)
        self.error_counts[severity] += 1
        
        return std_error
    
    def handle_validation_error(
        self,
        message: str,
        suggested_action: str = None,
        technical_details: str = None
    ) -> StandardizedError:
        """Handle validation errors specifically."""
        
        std_error = StandardizedError(
            message=message,
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.VALIDATION,
            technical_details=technical_details,
            suggested_action=suggested_action
        )
        
        self._log_error(std_error)
        self._show_user_feedback(std_error)
        self.error_history.append(std_error)
        self.error_counts[ErrorSeverity.WARNING] += 1
        
        return std_error
    
    def handle_data_error(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        suggested_action: str = None
    ) -> StandardizedError:
        """Handle data-related errors specifically."""
        
        std_error = StandardizedError(
            message=message,
            severity=severity,
            category=ErrorCategory.DATA_PROCESSING,
            suggested_action=suggested_action
        )
        
        self._log_error(std_error)
        self._show_user_feedback(std_error)
        self.error_history.append(std_error)
        self.error_counts[severity] += 1
        
        return std_error
    
    def _log_error(self, error: StandardizedError):
        """Log error with appropriate level."""
        log_message = f"[{error.category.value.upper()}] {error.message}"
        
        if error.technical_details:
            log_message += f" | Details: {error.technical_details}"
        
        if error.severity == ErrorSeverity.INFO:
            logger.info(log_message)
        elif error.severity == ErrorSeverity.WARNING:
            logger.warning(log_message)
        elif error.severity == ErrorSeverity.ERROR:
            logger.error(log_message)
        elif error.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
    
    def _show_user_feedback(self, error: StandardizedError):
        """Show appropriate user feedback in Streamlit."""
        feedback_message = error.message
        
        if error.suggested_action:
            feedback_message += f"\\n\\n💡 **Suggested action:** {error.suggested_action}"
        
        if error.severity == ErrorSeverity.INFO:
            st.info(feedback_message)
        elif error.severity == ErrorSeverity.WARNING:
            st.warning(feedback_message)
        elif error.severity == ErrorSeverity.ERROR:
            st.error(feedback_message)
        elif error.severity == ErrorSeverity.CRITICAL:
            st.error(f"🚨 **Critical Error:** {feedback_message}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        return {
            'total_errors': len(self.error_history),
            'error_counts': {k.value: v for k, v in self.error_counts.items()},
            'recent_errors': [error.to_dict() for error in self.error_history[-5:]],
            'categories': {}
        }
    
    def clear_errors(self):
        """Clear error history."""
        self.error_history = []
        self.error_counts = {severity: 0 for severity in ErrorSeverity}

# Global error handler instance
_global_error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler

def safe_execute(
    func,
    error_message: str = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    suggested_action: str = None,
    show_technical_details: bool = False,
    default_return=None
):
    """Decorator for safe function execution with standardized error handling."""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = get_error_handler()
                error_handler.handle_error(
                    e,
                    message=error_message or f"Error in {func.__name__}",
                    severity=severity,
                    category=category,
                    suggested_action=suggested_action,
                    show_technical_details=show_technical_details
                )
                return default_return
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

def handle_error(
    error: Exception,
    message: str = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    suggested_action: str = None,
    show_technical_details: bool = False
) -> StandardizedError:
    """Quick function to handle errors using the global handler."""
    return _global_error_handler.handle_error(
        error, message, severity, category, suggested_action, show_technical_details
    )

def handle_validation_error(
    message: str,
    suggested_action: str = None,
    technical_details: str = None
) -> StandardizedError:
    """Quick function to handle validation errors."""
    return _global_error_handler.handle_validation_error(
        message, suggested_action, technical_details
    )

def handle_data_error(
    message: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    suggested_action: str = None
) -> StandardizedError:
    """Quick function to handle data errors."""
    return _global_error_handler.handle_data_error(message, severity, suggested_action)

def error_dashboard():
    """Streamlit dashboard for error monitoring."""
    st.subheader("🚨 Error Monitoring Dashboard")
    
    error_handler = get_error_handler()
    summary = error_handler.get_error_summary()
    
    if summary['total_errors'] == 0:
        st.success("✅ No errors recorded in this session!")
        return
    
    # Error summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Errors", summary['total_errors'])
    with col2:
        st.metric("Critical", summary['error_counts'].get('critical', 0))
    with col3:
        st.metric("Errors", summary['error_counts'].get('error', 0))
    with col4:
        st.metric("Warnings", summary['error_counts'].get('warning', 0))
    
    # Recent errors
    if summary['recent_errors']:
        st.subheader("Recent Errors")
        for error_dict in summary['recent_errors']:
            severity = error_dict['severity']
            category = error_dict['category']
            message = error_dict['message']
            timestamp = error_dict['timestamp']
            
            # Choose appropriate display method
            if severity == 'critical':
                st.error(f"🚨 **[{category.upper()}]** {message} | {timestamp}")
            elif severity == 'error':
                st.error(f"❌ **[{category.upper()}]** {message} | {timestamp}")
            elif severity == 'warning':
                st.warning(f"⚠️ **[{category.upper()}]** {message} | {timestamp}")
            else:
                st.info(f"ℹ️ **[{category.upper()}]** {message} | {timestamp}")
    
    # Clear errors button
    if st.button("🗑️ Clear Error History"):
        error_handler.clear_errors()
        st.success("Error history cleared!")
        st.rerun()

# Context manager for error handling
class ErrorContext:
    """Context manager for handling errors in code blocks."""
    
    def __init__(
        self,
        message: str = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        suggested_action: str = None,
        show_technical_details: bool = False,
        reraise: bool = False
    ):
        self.message = message
        self.severity = severity
        self.category = category
        self.suggested_action = suggested_action
        self.show_technical_details = show_technical_details
        self.reraise = reraise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            handle_error(
                exc_val,
                self.message,
                self.severity,
                self.category,
                self.suggested_action,
                self.show_technical_details
            )
            return not self.reraise  # Suppress exception if reraise=False
        return False
