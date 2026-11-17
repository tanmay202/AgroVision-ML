@echo off
REM ============================================================
REM AgroVision Offline Demo - One-Click Launcher (Windows)
REM Requires Docker Desktop installed and running.
REM ============================================================

echo.
echo  =========================================
echo   AgroVision - Offline Demo Launcher
echo  =========================================
echo.

REM 1. Load Docker images from the tarball
echo [1/3] Loading Docker images (this may take a few minutes)...
docker load -i agrovision_demo.tar
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to load Docker images. Is Docker Desktop running?
    pause
    exit /b 1
)
echo       Images loaded successfully.
echo.

REM 2. Bring up the services
echo [2/3] Starting AgroVision containers...
docker compose up -d
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to start containers. Check docker-compose.yml is in the same folder.
    pause
    exit /b 1
)
echo       Containers started.
echo.

REM 3. Wait for health check
echo [3/3] Waiting for database and API to initialize...
timeout /t 15 /nobreak >nul

echo.
echo  =========================================
echo   AgroVision is running!
echo   Open http://localhost:8000 in your browser
echo  =========================================
echo.
echo  API Endpoints:
echo    Health:   http://localhost:8000/api/health
echo    Markets:  http://localhost:8000/api/markets
echo    Forecast: http://localhost:8000/api/forecast?mandi_id=WB_MND_CHAMPADANGA^&crop_id=POTATO_JYOTI
echo.
echo  To stop: docker compose down
echo.
pause
