@echo off
echo ====================================
echo  資源法 AI 戰情室 v2.1 啟動中...
echo ====================================
echo.
pip install -r requirements.txt
echo.
echo 啟動 Streamlit...
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
