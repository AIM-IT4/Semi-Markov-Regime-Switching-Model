
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic stock data with underlying regimes
def generate_stock_data(n_days=1000):
    # Define regimes: low volatility, medium volatility, and high volatility
    volatilities = [0.005, 0.015, 0.025]
    
    # Transition matrix for regimes (assume Markov switching for simplicity)
    transition_matrix = np.array([
        [0.98, 0.01, 0.01],
        [0.01, 0.98, 0.01],
        [0.01, 0.01, 0.98]
    ])
    
    # Initialize stock prices and regime list
    prices = [100]
    regimes = [1]  # Start in the medium volatility regime
    
    for i in range(1, n_days):
        # Determine regime for the day
        regime = np.random.choice([0, 1, 2], p=transition_matrix[regimes[-1]])
        regimes.append(regime)
        
        # Generate stock price based on regime's volatility
        daily_return = np.random.randn() * volatilities[regime]
        prices.append(prices[-1] * (1 + daily_return))
    
    return pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=n_days, freq='D'),
        'Price': prices,
        'Regime': regimes
    })

# Generate the data
stock_data = generate_stock_data()

# Plot the generated stock data and highlight the regimes
plt.figure(figsize=(15, 6))
for regime, color in zip([0, 1, 2], ['green', 'blue', 'red']):
    subset = stock_data[stock_data['Regime'] == regime]
    plt.plot(subset['Date'], subset['Price'], color=color)

plt.title('Synthetic Stock Data with Regimes')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(['Low Volatility', 'Medium Volatility', 'High Volatility'])
plt.grid(True)
plt.tight_layout()
plt.show()

# Simulate basic option strategy based on regimes
def simulate_strategy(stock_data):
    # Parameters
    option_premium_buy = 2  # Premium for buying an option
    option_premium_sell = 2.5  # Premium for selling an option (higher due to high volatility)
    
    # Initialize portfolio value and cash
    portfolio_values = [100000]  # Starting with 100,000 currency units
    cash = 100000
    
    for i in range(1, len(stock_data)):
        previous_price = stock_data['Price'].iloc[i - 1]
        current_price = stock_data['Price'].iloc[i]
        
        # Determine strategy based on regime
        if stock_data['Regime'].iloc[i] == 0:  # Low volatility
            cash -= option_premium_buy
            # Assume option gives a payoff if price goes up (for simplicity)
            if current_price > previous_price:
                cash += previous_price * 0.03  # 3% of the previous price as payoff
        elif stock_data['Regime'].iloc[i] == 2:  # High volatility
            cash += option_premium_sell
            # Assume option gives a payoff if price goes down (for simplicity)
            if current_price < previous_price:
                cash -= previous_price * 0.03  # 3% of the previous price as payoff
        
        # Assume rest of the money is invested in stock, calculate portfolio value
        portfolio_value = cash + (cash / previous_price) * current_price
        portfolio_values.append(portfolio_value)
    
    return portfolio_values

# Simulate the strategy
portfolio_values = simulate_strategy(stock_data)

# Plot portfolio performance
plt.figure(figsize=(15, 6))
plt.plot(stock_data['Date'], portfolio_values, label="Portfolio Value", color='purple')
plt.title('Portfolio Performance with Option Strategy')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
# Calculate strategy performance metrics

# Calculate daily returns
daily_returns = [0] + [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                       for i in range(1, len(portfolio_values))]

# Total Return
total_return = (portfolio_values[-1] / portfolio_values[0]) - 1

# Volatility (Annualized)
annualized_volatility = np.std(daily_returns) * np.sqrt(252)

# Sharpe Ratio (assuming a risk-free rate of 2%)
risk_free_rate = 0.02
average_daily_return = np.mean(daily_returns)
sharpe_ratio = (average_daily_return - risk_free_rate / 252) / np.std(daily_returns)

total_return, annualized_volatility, sharpe_ratio

# Main execution
if __name__ == "__main__":
    stock_data = generate_stock_data()
    portfolio_values = simulate_strategy(stock_data)
    total_return, annualized_volatility, sharpe_ratio = calculate_metrics(portfolio_values)
    print(f"Total Return: {total_return}")
    print(f"Annualized Volatility: {annualized_volatility}")
    print(f"Sharpe Ratio: {sharpe_ratio}")

