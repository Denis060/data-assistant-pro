"""
Advanced Performance Analytics Dashboard
Real-time system monitoring and performance optimization
"""

import psutil
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Any, Optional
import threading
import queue
import logging

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Advanced system performance monitoring and analytics
    """
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metrics_history = {
            'timestamp': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used_gb': [],
            'disk_usage_percent': [],
            'network_sent_mb': [],
            'network_recv_mb': [],
            'active_connections': [],
            'process_count': []
        }
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_queue = queue.Queue()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Python process specific
            current_process = psutil.Process()
            python_memory = current_process.memory_info()
            
            return {
                'timestamp': datetime.now(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency': cpu_freq.current if cpu_freq else None
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent': memory.percent,
                    'available_gb': memory.available / (1024**3)
                },
                'swap': {
                    'total_gb': swap.total / (1024**3),
                    'used_gb': swap.used / (1024**3),
                    'percent': swap.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'processes': {
                    'count': process_count,
                    'python_memory_mb': python_memory.rss / (1024**2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            return {}
    
    def start_monitoring(self, interval: float = 2.0):
        """Start real-time monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self, interval: float):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self.get_current_metrics()
                if metrics:
                    self.metrics_queue.put(metrics)
                    self._update_history(metrics)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(interval)
    
    def _update_history(self, metrics: Dict[str, Any]):
        """Update metrics history"""
        if len(self.metrics_history['timestamp']) >= self.history_size:
            # Remove oldest entries
            for key in self.metrics_history:
                self.metrics_history[key].pop(0)
        
        # Add new metrics
        self.metrics_history['timestamp'].append(metrics['timestamp'])
        self.metrics_history['cpu_percent'].append(metrics['cpu']['percent'])
        self.metrics_history['memory_percent'].append(metrics['memory']['percent'])
        self.metrics_history['memory_used_gb'].append(metrics['memory']['used_gb'])
        self.metrics_history['disk_usage_percent'].append(metrics['disk']['percent'])
        self.metrics_history['process_count'].append(metrics['processes']['count'])
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and recommendations"""
        current = self.get_current_metrics()
        if not current:
            return {}
        
        # Calculate performance scores
        cpu_score = max(0, 100 - current['cpu']['percent'])
        memory_score = max(0, 100 - current['memory']['percent'])
        disk_score = max(0, 100 - current['disk']['percent'])
        
        overall_score = (cpu_score + memory_score + disk_score) / 3
        
        # Generate recommendations
        recommendations = []
        if current['cpu']['percent'] > 80:
            recommendations.append("High CPU usage detected - consider optimizing algorithms")
        if current['memory']['percent'] > 85:
            recommendations.append("High memory usage - consider data chunking or cleanup")
        if current['disk']['percent'] > 90:
            recommendations.append("Low disk space - cleanup recommended")
        
        # Performance grade
        if overall_score >= 90:
            grade = "A"
        elif overall_score >= 80:
            grade = "B"
        elif overall_score >= 70:
            grade = "C"
        else:
            grade = "D"
        
        return {
            'overall_score': overall_score,
            'grade': grade,
            'cpu_score': cpu_score,
            'memory_score': memory_score,
            'disk_score': disk_score,
            'recommendations': recommendations,
            'current_metrics': current
        }
    
    def create_performance_charts(self) -> Dict[str, go.Figure]:
        """Create performance visualization charts"""
        if len(self.metrics_history['timestamp']) < 2:
            return {}
        
        charts = {}
        
        # CPU and Memory Usage Over Time
        fig_system = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CPU Usage %', 'Memory Usage %'),
            vertical_spacing=0.1
        )
        
        fig_system.add_trace(
            go.Scatter(
                x=self.metrics_history['timestamp'],
                y=self.metrics_history['cpu_percent'],
                mode='lines+markers',
                name='CPU %',
                line=dict(color='#ff6b6b', width=2)
            ),
            row=1, col=1
        )
        
        fig_system.add_trace(
            go.Scatter(
                x=self.metrics_history['timestamp'],
                y=self.metrics_history['memory_percent'],
                mode='lines+markers',
                name='Memory %',
                line=dict(color='#4ecdc4', width=2)
            ),
            row=2, col=1
        )
        
        fig_system.update_layout(
            title="System Performance Over Time",
            height=500,
            showlegend=False
        )
        
        charts['system_performance'] = fig_system
        
        # Resource Utilization Gauge
        current = self.get_current_metrics()
        if current:
            fig_gauges = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage')
            )
            
            # CPU Gauge
            fig_gauges.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=current['cpu']['percent'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "CPU %"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Memory Gauge
            fig_gauges.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=current['memory']['percent'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Memory %"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "green"},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 85], 'color': "yellow"},
                            {'range': [85, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=1, col=2
            )
            
            # Disk Gauge
            fig_gauges.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=current['disk']['percent'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Disk %"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "purple"},
                        'steps': [
                            {'range': [0, 70], 'color': "lightgray"},
                            {'range': [70, 90], 'color': "yellow"},
                            {'range': [90, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 95
                        }
                    }
                ),
                row=1, col=3
            )
            
            fig_gauges.update_layout(
                title="Current Resource Utilization",
                height=400
            )
            
            charts['resource_gauges'] = fig_gauges
        
        return charts
    
    def get_optimization_suggestions(self) -> List[Dict[str, str]]:
        """Get AI-powered optimization suggestions"""
        current = self.get_current_metrics()
        if not current:
            return []
        
        suggestions = []
        
        # Memory optimization
        if current['memory']['percent'] > 70:
            suggestions.append({
                'category': 'Memory Optimization',
                'suggestion': 'Use data chunking for large datasets',
                'code': 'df_chunks = pd.read_csv("large_file.csv", chunksize=10000)',
                'impact': 'High'
            })
            
            suggestions.append({
                'category': 'Memory Optimization',
                'suggestion': 'Optimize data types to reduce memory usage',
                'code': 'df = df.astype({"int_col": "int32", "float_col": "float32"})',
                'impact': 'Medium'
            })
        
        # CPU optimization
        if current['cpu']['percent'] > 60:
            suggestions.append({
                'category': 'CPU Optimization',
                'suggestion': 'Use vectorized operations instead of loops',
                'code': 'df["new_col"] = df["col1"] * df["col2"]  # Instead of apply()',
                'impact': 'High'
            })
            
            suggestions.append({
                'category': 'CPU Optimization',
                'suggestion': 'Consider parallel processing for heavy computations',
                'code': 'from multiprocessing import Pool; pool.map(function, data_chunks)',
                'impact': 'High'
            })
        
        # Disk optimization
        if current['disk']['percent'] > 80:
            suggestions.append({
                'category': 'Disk Optimization',
                'suggestion': 'Use efficient file formats like Parquet',
                'code': 'df.to_parquet("data.parquet")  # Smaller and faster',
                'impact': 'Medium'
            })
        
        return suggestions

def create_performance_dashboard():
    """
    Create advanced performance monitoring dashboard for Streamlit
    """
    st.header("⚡ Advanced Performance Analytics")
    
    # Initialize performance monitor
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
    
    monitor = st.session_state.performance_monitor
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Start Real-time Monitoring"):
            monitor.start_monitoring()
            st.success("Monitoring started!")
    
    with col2:
        if st.button("⏹️ Stop Monitoring"):
            monitor.stop_monitoring()
            st.info("Monitoring stopped!")
    
    with col3:
        if st.button("🔄 Refresh Metrics"):
            st.rerun()
    
    # Performance summary
    summary = monitor.get_performance_summary()
    if summary:
        st.subheader("📊 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Overall Score", f"{summary['overall_score']:.1f}/100")
        with col2:
            st.metric("Performance Grade", summary['grade'])
        with col3:
            st.metric("CPU Score", f"{summary['cpu_score']:.1f}/100")
        with col4:
            st.metric("Memory Score", f"{summary['memory_score']:.1f}/100")
        
        # Current metrics
        current = summary['current_metrics']
        
        # Resource utilization
        st.subheader("🔧 Current Resource Utilization")
        charts = monitor.create_performance_charts()
        
        if 'resource_gauges' in charts:
            st.plotly_chart(charts['resource_gauges'], use_container_width=True)
        
        # Historical performance
        if 'system_performance' in charts:
            st.subheader("📈 Performance History")
            st.plotly_chart(charts['system_performance'], use_container_width=True)
        
        # Detailed metrics
        with st.expander("📋 Detailed System Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**💻 CPU Information**")
                st.write(f"• Usage: {current['cpu']['percent']:.1f}%")
                st.write(f"• Cores: {current['cpu']['count']}")
                if current['cpu']['frequency']:
                    st.write(f"• Frequency: {current['cpu']['frequency']:.0f} MHz")
                
                st.write("**💾 Memory Information**")
                st.write(f"• Total: {current['memory']['total_gb']:.1f} GB")
                st.write(f"• Used: {current['memory']['used_gb']:.1f} GB")
                st.write(f"• Available: {current['memory']['available_gb']:.1f} GB")
                st.write(f"• Usage: {current['memory']['percent']:.1f}%")
            
            with col2:
                st.write("**💿 Disk Information**")
                st.write(f"• Total: {current['disk']['total_gb']:.1f} GB")
                st.write(f"• Used: {current['disk']['used_gb']:.1f} GB")
                st.write(f"• Usage: {current['disk']['percent']:.1f}%")
                
                st.write("**🔧 Process Information**")
                st.write(f"• Total processes: {current['processes']['count']}")
                st.write(f"• Python memory: {current['processes']['python_memory_mb']:.1f} MB")
        
        # Recommendations
        if summary['recommendations']:
            st.subheader("💡 Performance Recommendations")
            for rec in summary['recommendations']:
                st.warning(f"⚠️ {rec}")
        
        # Optimization suggestions
        st.subheader("🚀 AI-Powered Optimization Suggestions")
        suggestions = monitor.get_optimization_suggestions()
        
        if suggestions:
            for suggestion in suggestions:
                with st.expander(f"🔧 {suggestion['category']} - {suggestion['impact']} Impact"):
                    st.write(f"**Suggestion:** {suggestion['suggestion']}")
                    st.code(suggestion['code'], language='python')
        else:
            st.success("🎉 System is running optimally! No specific optimizations needed.")
    
    else:
        st.error("Unable to collect performance metrics. Please check system permissions.")

if __name__ == "__main__":
    # Test the performance monitor
    monitor = PerformanceMonitor()
    metrics = monitor.get_current_metrics()
    print(f"Current CPU: {metrics['cpu']['percent']:.1f}%")
    print(f"Current Memory: {metrics['memory']['percent']:.1f}%")
