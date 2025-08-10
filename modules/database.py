"""
Database Integration Module
Support for multiple database connections and data loading
"""

import pandas as pd
import streamlit as st
import logging
from typing import Dict, List, Optional, Tuple, Any
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
import sqlite3
import os
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database connections and operations."""
    
    def __init__(self):
        self.connections = {}
        self.current_connection = None
        
    def create_connection(self, 
                         db_type: str, 
                         connection_params: Dict[str, Any]) -> bool:
        """Create a database connection."""
        
        try:
            if db_type.lower() == 'sqlite':
                # SQLite connection
                db_path = connection_params.get('database', 'data.db')
                engine = create_engine(f'sqlite:///{db_path}')
                
            elif db_type.lower() == 'postgresql':
                # PostgreSQL connection
                host = connection_params.get('host', 'localhost')
                port = connection_params.get('port', 5432)
                database = connection_params.get('database')
                username = connection_params.get('username')
                password = quote_plus(connection_params.get('password', ''))
                
                engine = create_engine(
                    f'postgresql://{username}:{password}@{host}:{port}/{database}'
                )
                
            elif db_type.lower() == 'mysql':
                # MySQL connection
                host = connection_params.get('host', 'localhost')
                port = connection_params.get('port', 3306)
                database = connection_params.get('database')
                username = connection_params.get('username')
                password = quote_plus(connection_params.get('password', ''))
                
                engine = create_engine(
                    f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
                )
                
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
            
            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            connection_name = f"{db_type}_{connection_params.get('database', 'default')}"
            self.connections[connection_name] = {
                'engine': engine,
                'type': db_type,
                'params': connection_params
            }
            self.current_connection = connection_name
            
            logger.info(f"Successfully connected to {db_type} database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            st.error(f"Database connection failed: {str(e)}")
            return False
    
    def get_tables(self) -> List[str]:
        """Get list of tables in the current database."""
        
        if not self.current_connection:
            return []
        
        try:
            engine = self.connections[self.current_connection]['engine']
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            logger.info(f"Found {len(tables)} tables in database")
            return tables
            
        except Exception as e:
            logger.error(f"Failed to get tables: {str(e)}")
            return []
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get detailed information about a table."""
        
        if not self.current_connection:
            return {}
        
        try:
            engine = self.connections[self.current_connection]['engine']
            inspector = inspect(engine)
            
            columns = inspector.get_columns(table_name)
            row_count = self.get_row_count(table_name)
            
            return {
                'columns': columns,
                'row_count': row_count,
                'column_names': [col['name'] for col in columns],
                'column_types': {col['name']: str(col['type']) for col in columns}
            }
            
        except Exception as e:
            logger.error(f"Failed to get table info: {str(e)}")
            return {}
    
    def get_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        
        try:
            engine = self.connections[self.current_connection]['engine']
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.scalar()
                
        except Exception as e:
            logger.error(f"Failed to get row count: {str(e)}")
            return 0
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        
        if not self.current_connection:
            raise ValueError("No active database connection")
        
        try:
            engine = self.connections[self.current_connection]['engine']
            df = pd.read_sql_query(query, engine)
            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise e
    
    def load_table(self, 
                   table_name: str, 
                   limit: Optional[int] = None,
                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Load data from a table."""
        
        # Build query
        if columns:
            column_str = ', '.join(columns)
        else:
            column_str = '*'
        
        query = f"SELECT {column_str} FROM {table_name}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query)
    
    def save_dataframe(self, 
                      df: pd.DataFrame, 
                      table_name: str, 
                      if_exists: str = 'replace') -> bool:
        """Save DataFrame to database table."""
        
        if not self.current_connection:
            raise ValueError("No active database connection")
        
        try:
            engine = self.connections[self.current_connection]['engine']
            df.to_sql(table_name, engine, if_exists=if_exists, index=False)
            logger.info(f"Successfully saved DataFrame to table '{table_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save DataFrame: {str(e)}")
            return False


def create_sample_database():
    """Create a sample SQLite database for demonstration."""
    
    db_path = "sample_data.db"
    
    # Create sample data
    employees = pd.DataFrame({
        'id': range(1, 101),
        'name': [f'Employee_{i}' for i in range(1, 101)],
        'department': ['Sales', 'Engineering', 'HR', 'Marketing'] * 25,
        'salary': [50000 + i * 1000 for i in range(100)],
        'hire_date': pd.date_range('2020-01-01', periods=100, freq='D')
    })
    
    sales = pd.DataFrame({
        'id': range(1, 201),
        'employee_id': [i % 100 + 1 for i in range(200)],
        'amount': [1000 + i * 50 for i in range(200)],
        'sale_date': pd.date_range('2023-01-01', periods=200, freq='D')
    })
    
    # Save to SQLite
    engine = create_engine(f'sqlite:///{db_path}')
    employees.to_sql('employees', engine, if_exists='replace', index=False)
    sales.to_sql('sales', engine, if_exists='replace', index=False)
    
    return db_path


def database_dashboard():
    """Streamlit dashboard for database operations."""
    
    st.subheader("🗃️ Database Integration")
    
    # Initialize database manager
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    db_manager = st.session_state.db_manager
    
    # Database connection section
    with st.expander("🔌 Database Connection", expanded=True):
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            db_type = st.selectbox(
                "Database Type",
                ['SQLite', 'PostgreSQL', 'MySQL'],
                help="Select your database type"
            )
        
        with col2:
            if st.button("📊 Create Sample Database"):
                with st.spinner("Creating sample database..."):
                    db_path = create_sample_database()
                    st.success(f"✅ Sample database created: {db_path}")
        
        # Connection parameters based on database type
        if db_type == 'SQLite':
            database = st.text_input(
                "Database File", 
                value="sample_data.db",
                help="Path to SQLite database file"
            )
            connection_params = {'database': database}
            
        elif db_type == 'PostgreSQL':
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host", value="localhost")
                database = st.text_input("Database Name")
            with col2:
                port = st.number_input("Port", value=5432)
                username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            connection_params = {
                'host': host, 'port': port, 'database': database,
                'username': username, 'password': password
            }
            
        else:  # MySQL
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host", value="localhost")
                database = st.text_input("Database Name")
            with col2:
                port = st.number_input("Port", value=3306)
                username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            connection_params = {
                'host': host, 'port': port, 'database': database,
                'username': username, 'password': password
            }
        
        # Connect button
        if st.button("🔌 Connect to Database"):
            with st.spinner("Connecting..."):
                success = db_manager.create_connection(db_type, connection_params)
                if success:
                    st.success("✅ Connected successfully!")
                    st.rerun()
    
    # Show connection status
    if db_manager.current_connection:
        st.success(f"✅ Connected to: {db_manager.current_connection}")
        
        # Table operations section
        with st.expander("📋 Database Tables", expanded=True):
            
            tables = db_manager.get_tables()
            
            if tables:
                selected_table = st.selectbox("Select Table", tables)
                
                if selected_table:
                    # Table information
                    table_info = db_manager.get_table_info(selected_table)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", f"{table_info.get('row_count', 0):,}")
                    with col2:
                        st.metric("Columns", len(table_info.get('columns', [])))
                    with col3:
                        if st.button("📊 Load Table"):
                            with st.spinner("Loading data..."):
                                df = db_manager.load_table(selected_table, limit=1000)
                                st.session_state.df = df
                                st.session_state.cleaned_df = df.copy()
                                st.success(f"✅ Loaded {len(df)} rows from {selected_table}")
                                st.rerun()
                    
                    # Show table schema
                    if table_info.get('columns'):
                        st.subheader("📋 Table Schema")
                        schema_df = pd.DataFrame(table_info['columns'])
                        st.dataframe(schema_df, use_container_width=True)
            
            else:
                st.info("No tables found in the database")
        
        # Custom SQL query section
        with st.expander("🔍 Custom SQL Query"):
            
            query = st.text_area(
                "SQL Query",
                value="SELECT * FROM employees LIMIT 10;",
                height=100,
                help="Enter your custom SQL query"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("▶️ Execute"):
                    try:
                        with st.spinner("Executing query..."):
                            result_df = db_manager.execute_query(query)
                            st.session_state.query_result = result_df
                            st.success(f"✅ Query executed! {len(result_df)} rows returned")
                    except Exception as e:
                        st.error(f"❌ Query failed: {str(e)}")
            
            # Show query results
            if 'query_result' in st.session_state:
                st.subheader("📊 Query Results")
                st.dataframe(st.session_state.query_result, use_container_width=True)
                
                if st.button("📥 Use Query Result as Dataset"):
                    st.session_state.df = st.session_state.query_result
                    st.session_state.cleaned_df = st.session_state.query_result.copy()
                    st.success("✅ Query result loaded as current dataset!")
                    st.rerun()
        
        # Save data to database
        if 'cleaned_df' in st.session_state:
            with st.expander("💾 Save Data to Database"):
                
                new_table_name = st.text_input(
                    "Table Name", 
                    value="cleaned_data",
                    help="Name for the new table"
                )
                
                if_exists = st.selectbox(
                    "If Table Exists",
                    ['replace', 'append', 'fail'],
                    help="What to do if table already exists"
                )
                
                if st.button("💾 Save Cleaned Data"):
                    with st.spinner("Saving data..."):
                        success = db_manager.save_dataframe(
                            st.session_state.cleaned_df, 
                            new_table_name, 
                            if_exists=if_exists
                        )
                        if success:
                            st.success(f"✅ Data saved to table '{new_table_name}'")
                        else:
                            st.error("❌ Failed to save data")
    
    else:
        st.info("👆 Please connect to a database first")


# Query templates for common operations
QUERY_TEMPLATES = {
    "Show all tables": "SELECT name FROM sqlite_master WHERE type='table';",
    "Count rows": "SELECT COUNT(*) FROM {table_name};",
    "Show sample data": "SELECT * FROM {table_name} LIMIT 10;",
    "Column info": "PRAGMA table_info({table_name});",
    "Find duplicates": "SELECT *, COUNT(*) FROM {table_name} GROUP BY {column} HAVING COUNT(*) > 1;",
    "Date range": "SELECT * FROM {table_name} WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}';",
    "Top values": "SELECT {column}, COUNT(*) as count FROM {table_name} GROUP BY {column} ORDER BY count DESC LIMIT 10;",
}
