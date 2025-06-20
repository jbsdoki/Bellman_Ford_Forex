import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta

# Test data with a known profitable cycle: USD -> EUR -> GBP -> USD
# Starting with 100 USD:
# 100 USD -> 95 EUR (rate: 0.95)
# 95 EUR -> 100 GBP (rate: 1.0526)
# 100 GBP -> 105 USD (rate: 1.05)
# Final amount: 105 USD (5% profit)

# Create test data
test_data = {
    'USDEUR=X': [0.95],  # 1 USD = 0.95 EUR
    'EURGBP=X': [1.0526],  # 1 EUR = 1.0526 GBP
    'GBPUSD=X': [1.05],  # 1 GBP = 1.05 USD
    'EURUSD=X': [1/0.95],  # 1 EUR = 1.0526 USD
    'GBPEUR=X': [1/1.0526],  # 1 GBP = 0.95 EUR
    'USDGBP=X': [1/1.05],  # 1 USD = 0.9524 GBP
}

# Create DataFrame with test data
test_df = pd.DataFrame(test_data, index=[datetime.now()])

def test_bellman_ford_original():
    """Test the original Bellman-Ford implementation"""
    from Bellman_Ford_Algo_For_CX import check_arbitrage_for_day
    
    print("\nTesting original Bellman-Ford implementation...")
    result = check_arbitrage_for_day(test_df.iloc[0], test_df.index[0])
    print(f"Found arbitrage: {result}")

def test_networkx_bellman_ford():
    """Test the NetworkX Bellman-Ford implementation"""
    from networkX_BF_Algo_For_CX import check_arbitrage_for_day
    
    print("\nTesting NetworkX Bellman-Ford implementation...")
    result = check_arbitrage_for_day(test_df.iloc[0], test_df.index[0])
    print(f"Found arbitrage: {result}")

def verify_cycle_manually():
    """Manually verify the test cycle"""
    print("\nManually verifying the test cycle:")
    amount = 100  # Start with 100 USD
    print(f"Starting with {amount} USD")
    
    # USD -> EUR
    amount *= 0.95
    print(f"USD -> EUR: {amount:.2f} EUR (rate: 0.95)")
    
    # EUR -> GBP
    amount *= 1.0526
    print(f"EUR -> GBP: {amount:.2f} GBP (rate: 1.0526)")
    
    # GBP -> USD
    amount *= 1.05
    print(f"GBP -> USD: {amount:.2f} USD (rate: 1.05)")
    
    profit = amount - 100
    profit_percentage = (profit / 100) * 100
    print(f"\nTotal profit: {profit:.2f} USD ({profit_percentage:.2f}%)")

if __name__ == "__main__":
    print("Testing currency arbitrage detection with known profitable cycle")
    print("Test cycle: USD -> EUR -> GBP -> USD (expected profit: 5%)")
    
    # Run all tests
    test_bellman_ford_original()
    test_networkx_bellman_ford()
    verify_cycle_manually() 