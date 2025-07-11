# AgriCast - Agricultural Commodity Price Forecasting

AgriCast is a hybrid machine learning system for forecasting agricultural commodity prices using ensemble methods that combine SARIMA, LSTM, and XGBoost models.

## Overview

**AgriCast** leverages multiple data sources and advanced machine learning techniques to predict agricultural commodity prices. The system integrates traditional time series analysis (SARIMA), deep learning (LSTM), and ensemble methods (XGBoost) to capture both linear and non-linear price patterns.

## Features

- **Multi-source Data Collection**: Automated scrapers for agricultural price data from multiple sources
- **Hybrid Forecasting Model**: Combines SARIMA, LSTM, and XGBoost for improved accuracy
- **Comprehensive Evaluation**: Metrics include MAE, RMSE, MAPE, and R²
- **Flexible Data Processing**: Handles missing data with interpolation and supports various time frequencies
- **Modular Architecture**: Clean separation of data collection, modeling, and evaluation


## Installation

1. **Clone the repository:**
    ```
    git clone https://github.com/PrakshaaleJain/AgriCast.git
    cd AgriCast
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**  
   Create a `.env` file with the following:
    ```
    commodity_ID_path=path/to/your/commodity_ids.csv
    ```

## Usage

### Data Collection

- **Scrape commodity and district IDs:**
    ```
    python ID_scraper/commodities_ids.py
    python ID_scraper/district_ids.py
    ```
- **Download historical price data:**
    ```
    python CEDA_scraper.py
    ```
- **Alternative web scraping:**
    ```
    python scraper.py
    ```

### Model Training

- **Run the hybrid forecasting pipeline:**
    ```
    python train.py
    ```

### Model Evaluation

- **Evaluate model performance:**
    ```
    python metric.py
    ```

## Model Architecture

| Model    | Purpose                                 | Details                          |
|----------|-----------------------------------------|----------------------------------|
| SARIMA   | Linear trends & seasonality             | Order (1,1,1), seasonal (1,1,1,12) |
| LSTM     | Non-linear pattern learning             | 2 layers, 64 hidden units        |
| XGBoost  | Ensemble of SARIMA and LSTM predictions | 400 estimators                   |

## Data Sources

- **CEDA (Centre for Economic Data and Analysis):** Primary source for commodity prices
- **FCA Info Web:** Alternative source for retail price data
- **Supported Commodities:** Wheat, Rice, and others
- **Geographic Coverage:** District-level data across India

## Dependencies

- pandas
- numpy
- matplotlib
- torch
- seaborn
- statsmodels
- scikit-learn
- xgboost
- scrapy
- selenium

## Model Performance Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License

## Future Enhancements

- Support for more commodities
- Real-time API for price prediction
- Weather data integration
- Mobile app for farmers
- Advanced feature engineering

## Contact
For questions or support, please open an issue on the GitHub repository.
