
# Regime-Switching Option Trading Strategy

This project presents a basic option trading strategy based on a regime-switching concept. The strategy is implemented using synthetic stock data with underlying volatility regimes.

## Data

The synthetic stock data has three underlying volatility regimes:
- Low Volatility
- Medium Volatility
- High Volatility

## Strategy

The trading strategy involves buying options during periods of low volatility and selling options during periods of high volatility.

## Performance Metrics

The performance of the strategy is evaluated using the following metrics:
- Total Return
- Annualized Volatility
- Sharpe Ratio

## Files
- `synthetic_stock_data.csv`: Contains the synthetic stock data with price and regime information.
- `trading_strategy.py`: Python script with the code for generating data, simulating the strategy, and calculating performance metrics.

## Note

This project is based on a conceptual discussion and uses synthetic data. It's essential to validate any trading strategy with real-world data and consider transaction costs, slippage, taxes, etc., before implementation.
