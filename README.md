NSE Stock Price Prediction Project
This Python project analyzes and predicts stock prices using a combination of machine learning and time series forecasting models on the India Stock Data (NSE 1990-2020) dataset from Kaggle.

Dataset
The dataset contains daily historical stock prices (Open, High, Low, Close) and volume for multiple NSE-listed companies from 1990 to 2020. Users must download the dataset from Kaggle, extract it, and save the extracted folder as Datasets on their Desktop:

text
C:\Users\<YourUserName>\Desktop\Datasets
The project code uses this path for accessing the data.

Features and Indicators Computed
Moving Averages (50-day and 200-day)

Relative Strength Index (RSI)

Moving Average Convergence Divergence (MACD)

Bollinger Bands

These indicators help capture trends and momentum for use in forecasting models.

Machine Learning Models Used
1. ARIMA (AutoRegressive Integrated Moving Average)
ARIMA is a classical time series forecasting method that models temporal autocorrelations in price data. It provides short-term forecasts based on past stock price patterns.
Output: Forecasted stock closing prices for a test period, capturing seasonality and trends.

2. Random Forest Regressor
A powerful ensemble model that uses multiple decision trees trained on technical indicators and historical prices. It captures nonlinear relationships and interactions between features.
Output: Predicted stock closing prices on the test set, showing robust performance on noisy and complex data.

3. Support Vector Regression (SVR)
SVR uses Support Vector Machines adapted for regression, modeling nonlinear dependencies with kernel functions. Suitable for smaller datasets or features with complex patterns.
Output: Predicted closing prices, offering a balance between bias and variance with good generalization.

Model Evaluation
Models are evaluated using:

Mean Absolute Error (MAE): average absolute difference between predicted and true prices.

Root Mean Squared Error (RMSE): penalizes larger errors, measuring prediction accuracy.

An ensemble forecast averaging the above models typically yields improved performance by combining their strengths.

Running the Project
Install dependencies:

bash
pip install -r requirements.txt
Run the analysis:

bash
python stocksproject.py
When prompted, select the stock index from the list to analyze and forecast.

Final Output
Printed model performance metrics (MAE, RMSE) for each forecasting model.

A unified plot showing historical price, moving averages, and test set predictions from all models and their ensemble.

