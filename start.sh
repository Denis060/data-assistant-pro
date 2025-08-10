#!/bin/bash

# Data Assistant Pro - Startup Script
# This script starts both the Streamlit app and FastAPI server

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Data Assistant Pro Startup Script ===${NC}"
echo -e "${YELLOW}Enterprise-grade data science platform${NC}"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi

# Check if required packages are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import streamlit, fastapi, uvicorn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing required packages...${NC}"
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install dependencies${NC}"
        exit 1
    fi
fi

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    if [ ! -z "$STREAMLIT_PID" ] && kill -0 $STREAMLIT_PID 2>/dev/null; then
        kill $STREAMLIT_PID
        echo -e "${GREEN}Streamlit app stopped${NC}"
    fi
    if [ ! -z "$API_PID" ] && kill -0 $API_PID 2>/dev/null; then
        kill $API_PID
        echo -e "${GREEN}API server stopped${NC}"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Get user preference for what to start
echo "What would you like to start?"
echo "1) Streamlit App only (port 8501)"
echo "2) API Server only (port 8000)" 
echo "3) Both Streamlit App and API Server"
echo "4) API Server with custom port"
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo -e "${GREEN}Starting Streamlit App...${NC}"
        echo -e "${BLUE}URL: http://localhost:8501${NC}"
        streamlit run app.py --server.port 8501 --server.address 0.0.0.0
        ;;
    2)
        echo -e "${GREEN}Starting API Server...${NC}"
        echo -e "${BLUE}URL: http://localhost:8000${NC}"
        echo -e "${BLUE}API Docs: http://localhost:8000/docs${NC}"
        python api_server.py
        ;;
    3)
        echo -e "${GREEN}Starting both services...${NC}"
        
        # Start API Server in background
        echo -e "${YELLOW}Starting API Server on port 8000...${NC}"
        python api_server.py &
        API_PID=$!
        sleep 3
        
        # Check if API server started successfully
        if ! kill -0 $API_PID 2>/dev/null; then
            echo -e "${RED}Failed to start API server${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}API Server started (PID: $API_PID)${NC}"
        echo -e "${BLUE}API URL: http://localhost:8000${NC}"
        echo -e "${BLUE}API Docs: http://localhost:8000/docs${NC}"
        
        # Start Streamlit app in background
        echo -e "${YELLOW}Starting Streamlit App on port 8501...${NC}"
        streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
        STREAMLIT_PID=$!
        sleep 3
        
        # Check if Streamlit started successfully
        if ! kill -0 $STREAMLIT_PID 2>/dev/null; then
            echo -e "${RED}Failed to start Streamlit app${NC}"
            cleanup
            exit 1
        fi
        
        echo -e "${GREEN}Streamlit App started (PID: $STREAMLIT_PID)${NC}"
        echo -e "${BLUE}Streamlit URL: http://localhost:8501${NC}"
        
        echo ""
        echo -e "${GREEN}=== Both services are running! ===${NC}"
        echo -e "${BLUE}Streamlit App: http://localhost:8501${NC}"
        echo -e "${BLUE}API Server: http://localhost:8000${NC}" 
        echo -e "${BLUE}API Documentation: http://localhost:8000/docs${NC}"
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
        
        # Wait for both processes
        wait
        ;;
    4)
        read -p "Enter API server port (default 8000): " api_port
        api_port=${api_port:-8000}
        echo -e "${GREEN}Starting API Server on port $api_port...${NC}"
        echo -e "${BLUE}URL: http://localhost:$api_port${NC}"
        echo -e "${BLUE}API Docs: http://localhost:$api_port/docs${NC}"
        uvicorn api_server:app --host 0.0.0.0 --port $api_port
        ;;
    *)
        echo -e "${RED}Invalid choice. Please run the script again.${NC}"
        exit 1
        ;;
esac
