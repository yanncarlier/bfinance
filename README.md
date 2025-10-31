



uv init  
uv run  main.py  
uv add yfinance  

source .venv/bin/activate
uv pip install ccxt pandas python-dotenv schedule

uv pip freeze > requirements.txt  



uv run trading_bot.py

uv pip install streamlit  

uv pip install plotly  

uv pip install numpy  



uv run streamlit run dashboard.py





code .
shift ctrl p
