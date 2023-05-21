# Big-Data-Analytics-Techniques-and-Applications-
NYCU 2023 BDA Final Project  
Exploring Predictive Factors for US Stock Movements

## Repository Structure
```bash
├── data # folder contains all training and testing data  
│   └── df_{stock}.csv  
├── data_preprocessing  
│   ├── data_preprocessing.ipynb  
│   └── df_all_stocks.csv   
└── results # folder contains different window size results  
    ├── Window_10.csv   
    ├── Window_20.csv  
    └── Window_5.csv  
└── model.py # model for generating results  
```

## Data
* Stock Price:
    - date: timestamp of the record
    - tic: ticket name
    - open:  the open price of the period
    - high: the highest price of the interval
    - low: the lowest price of the interval
    - close: the close price of the period
* Technical Indicators:
    - volume: the volume of stocks traded during the interval
    - day: The day of week
    - macd: Moving Average Convergence Divergence
    - boll_ub: Bollinger's Upper Bands
    - boll_lb: Bollinger's Low Bands
    - rsi_30: Relative Strength Index
    - cci_30: Commodity Channel Index
    - dx_30: Directional Index
    - close_30_sma: 30-day Closing Simple Moving Average
    - close_60_sma: 60-day Closing Simple Moving Average
* Macroeconomics Indicators:
    - EFFR: Effective Federal Funds Rate
    - DTB3: 3-Month Treasury Bill Secondary Market Rate
    - T10YIE: 10-Yaer Breakeven Inflation Rate
    - DGS10: Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity
    - GC=F: Gold Price
    - CL=F: Crude Oil Price

## Models
* Algorithms
    - Logistic Regression
    - Random forest Classifier
    - Gradient-Boosted Trees Classifier
    - Decision Tree Classifier
* Window size (days)
    - 5 days
    - 10 days
    - 20 days
* Data Combinations
    - Stock Price (S)
    - Stock Price + Techinical Indicators (S + T)
    - Stock Price + Macroeconomic Indicators (S + M)
    - Stock Price + Techinical Indicators + Macroeconomic Indicators (S + T + M)
* Usage
    ```bash
    python model.py
    ```
## Results
* Window_5.csv
* Window_10.csv
* Window_20.csv
