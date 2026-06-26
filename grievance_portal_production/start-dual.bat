@echo off
title Tamil Nadu Grievance Portal - Production Launcher

echo.
echo    🏛️ Tamil Nadu Grievance Portal - Production Ready
echo    ===============================================
echo    AI-Enhanced Classification • SMS Notifications • Location Services
echo    Opening Frontend and Backend in separate windows
echo.
echo 🐍 Recommended Python Version: 3.9.5
echo    (Compatible with all AI/ML dependencies)
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.9.5 first.
    echo 💡 Download from: https://www.python.org/downloads/release/python-395/
    pause
    exit /b 1
)

echo 📋 Checking Python version compatibility...
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Current Python Version: %PYTHON_VERSION%

REM Check if Python version is 3.9.x (recommended)
echo %PYTHON_VERSION% | findstr /R "^3\.9\." >nul
if not errorlevel 1 (
    echo ✅ Perfect! Python 3.9.x detected - fully compatible with all dependencies
) else (
    echo %PYTHON_VERSION% | findstr /R "^3\.[8-9]\|^3\.1[0-1]\." >nul
    if not errorlevel 1 (
        echo ⚠️  Python version compatible but not optimal
        echo 💡 For best compatibility, consider using Python 3.9.5
    ) else (
        echo ❌ Python version may cause dependency conflicts
        echo 🔧 Please install Python 3.9.5 for optimal compatibility
        echo 💡 Download from: https://www.python.org/downloads/release/python-395/
        echo.
        echo Press any key to continue anyway, or Ctrl+C to exit and install Python 3.9.5...
        pause
    )
)

echo.
echo 📦 Installing/Updating dependencies...
echo    Core dependencies: FastAPI, MongoDB, Groq AI, Twilio SMS
echo    Using Python %PYTHON_VERSION% package manager...
python -m pip install -r backend/requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies. Please check your internet connection.
    echo 💡 Try running: python -m pip install --upgrade pip
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully!

REM Check if .env file exists
if not exist "backend\.env" (
    echo.
    echo ⚠️  Warning: backend\.env file not found!
    echo Please create backend\.env with required environment variables:
    echo.
    echo Required variables:
    echo   MONGODB_URI=your_mongodb_connection_string
    echo   GROQ_API_KEY=your_groq_api_key
    echo.
    echo Optional SMS notifications:
    echo   TWILIO_ACCOUNT_SID=your_twilio_account_sid
    echo   TWILIO_AUTH_TOKEN=your_twilio_auth_token
    echo   TWILIO_PHONE_NUMBER=your_twilio_phone_number
    echo.
    echo Optional AI features:
    echo   OPENAI_API_KEY=your_openai_api_key
    echo.
    pause
    exit /b 1
)

REM Check for SMS service configuration
findstr /C:"TWILIO_ACCOUNT_SID" backend\.env >nul
if errorlevel 1 (
    echo.
    echo ℹ️  SMS notifications not configured ^(optional^)
    echo.
    echo 📱 To enable SMS notifications, add to backend\.env:
    echo   TWILIO_ACCOUNT_SID=your_twilio_account_sid
    echo   TWILIO_AUTH_TOKEN=your_twilio_auth_token  
    echo   TWILIO_PHONE_NUMBER=your_twilio_phone_number
    echo.
    echo 💡 Get Twilio credentials from: https://www.twilio.com/console
    echo.
)

REM Check for AI features (optional)
findstr /C:"OPENAI_API_KEY" backend\.env >nul
if errorlevel 1 (
    echo ℹ️  Advanced AI features not configured ^(optional^)
    echo 💡 System will use free AI alternatives automatically
    echo.
)

echo.
echo 🧪 Testing system services...
timeout /t 2 /nobreak >nul

echo.
echo 🌐 Opening frontend in browser...
timeout /t 1 /nobreak >nul
start "" "http://localhost:8000/frontend/index.html"

echo 🚀 Opening backend server in new terminal...
start "Tamil Nadu Grievance Portal - Backend" cmd /k "echo ============================================== && echo  🏛️ Tamil Nadu Grievance Portal Backend && echo  🐍 Python Version: %PYTHON_VERSION% && echo ============================================== && echo. && echo 🌐 Frontend: http://localhost:8000/frontend/index.html && echo 📖 API Docs: http://localhost:8000/docs && echo 🔧 Admin: http://localhost:8000/frontend/officer_dashboard.html && echo. && echo � Features: && echo    - RAG-Enhanced AI Classification && echo    - SMS Notifications via Twilio && echo    - Camera Location Capture && echo    - 37 Tamil Nadu Departments && echo. && echo 🛑 Press Ctrl+C to stop the server && echo ============================================== && echo. && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo.
echo ✅ Setup Complete!
echo.
echo 📱 Frontend: http://localhost:8000/frontend/index.html
echo 🖥️  Backend: Running in separate terminal window
echo 📖 API Docs: http://localhost:8000/docs
echo 🔧 Admin Dashboard: http://localhost:8000/frontend/officer_dashboard.html
echo.
echo �️ Production Features:
echo    🧠 RAG + Groq AI Classification (37 TN Departments)
echo    📱 SMS Notifications (Twilio Integration)
echo    📸 Camera Location Capture
echo    🎤 Audio/Image AI Analysis (Optional)
echo    🔒 Secure Admin Dashboard
echo.
echo 💡 Both windows are now running independently
echo 🛑 Close the backend terminal window to stop the server
echo.
echo � Complete Documentation: COMPLETE_PROJECT_GUIDE.md
echo 🐍 Python Version: %PYTHON_VERSION%
echo.
pause