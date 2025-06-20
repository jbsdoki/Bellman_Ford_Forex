# Extracting Data from Yahoo Finance with yFinance
# Eurico Paes
# https://medium.com/@euricopaes/extracting-data-from-yahoo-finance-with-yfinance-96798253d8ca

# Downloading Forex Price Data with yFinance Library in Python
# José Carlos Gonzáles Tanaka
# https://blog.quantinsti.com/download-forex-price-data-yfinance-library-python/

# Top currencies used
# https://www.xs.com/en/blog/strongest-currencies-in-the-world/


import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime, timedelta



# Define currencies and create pairs
currencies = ['KWD', 'BHD', 'OMR', 'JOD', 'GBP', 'GIP', 'FKP', 
              'KYD', 'CHF', 'EUR', 'USD', 'SGD', 'BND', 'CAD', 
              'AUD', 'AZN', 'NZD', 'AWG', 'BGN', 'BAM', 'BZD', 
              'BBD', 'FJD', 'TOP', 'GEL', 'XCD', 'QAR', 'SAR', 
              'AED', 'MYR', 'CNY', 'TRY', 'MXN', 'THB', 'ZAR']
pairs = [f"{a}{b}=X" for a in currencies for b in currencies if a != b]

# Get data from the last 2 years (730 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

# Download exchange rate data with error handling
print(f"Downloading data from {start_date.date()} to {end_date.date()}")
try:
    raw_data = yf.download(pairs, start=start_date, end=end_date, progress=False)
except Exception as e:
    print(f"Error downloading data: {str(e)}")
    # Try downloading in smaller batches
    raw_data = pd.DataFrame()
    batch_size = 10
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        try:
            batch_data = yf.download(batch, start=start_date, end=end_date, progress=False)
            if not batch_data.empty:
                raw_data = pd.concat([raw_data, batch_data], axis=1)
        except Exception as e:
            print(f"Error downloading batch {i//batch_size + 1}: {str(e)}")
            continue

# Get the Close prices and drop any columns with all NaN values
data = raw_data['Close'].dropna(axis=1, how='all')

print("\nAvailable currency pairs:")
print(data.columns.tolist())
print("\nSample exchange rates:")
print(data.iloc[-1].head())

def find_profitable_cycle(G, source):
    """
    Find any cycle starting and ending at source that results in a profit using NetworkX's Bellman-Ford.
    Returns the cycle if found, None otherwise.
    """
    try:
        # Get all pairs shortest path lengths using Bellman-Ford
        path_lengths = dict(nx.all_pairs_bellman_ford_path_length(G, weight='weight'))
        
        # Check each possible cycle length
        for target in G.nodes():
            if target == source:
                continue
                
            # Get the path from source to target
            try:
                path = nx.shortest_path(G, source, target, weight='weight')
                
                # Check if we can return to source with a profit
                if target in G[source]:
                    # Calculate the total weight of the cycle
                    cycle_weight = path_lengths[source][target] + G[target][source]['weight']
                    
                    # If the cycle weight is negative, we found a profitable cycle
                    if cycle_weight < 0:
                        # Complete the cycle by adding the source at the end
                        cycle = path + [source]
                        
                        # Calculate the final amount after following the cycle
                        amount = 100  # Start with 100 units
                        for i in range(len(cycle)-1):
                            rate = np.exp(-G[cycle[i]][cycle[i+1]]['weight'])
                            amount *= rate
                        
                        # If we end up with strictly more than 100 units, this is a profitable cycle
                        if amount > 100.001:  # Using a small epsilon to account for floating point precision
                            return cycle, amount
            except nx.NetworkXNoPath:
                continue
                
    except nx.NetworkXUnbounded:
        # This exception is raised when a negative cycle is detected
        # We can use this to find the cycle
        try:
            # Try to find a negative cycle using NetworkX's negative_edge_cycle
            cycle = nx.find_negative_edge_cycle(G, weight='weight')
            if cycle:
                # Calculate the final amount
                amount = 100
                for i in range(len(cycle)-1):
                    rate = np.exp(-G[cycle[i]][cycle[i+1]]['weight'])
                    amount *= rate
                return cycle, amount
        except:
            pass
    
    return None, None

def check_arbitrage_for_day(rates, date):
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges with raw exchange rates
    for pair, rate in rates.items():
        if pd.isna(rate):
            continue  # Skip missing rates

        base = pair[:3]
        quote = pair[3:6]

        try:
            # Add both directions of the exchange rate
            G.add_edge(base, quote, weight=-np.log(rate))
            G.add_edge(quote, base, weight=np.log(rate))
        except Exception as e:
            print(f"Skipping {pair} due to error: {e}")

    # Check for arbitrage opportunities starting from each currency
    for source in currencies:
        cycle, final_amount = find_profitable_cycle(G, source)
        if cycle and final_amount is not None:
            # Calculate profit percentage
            profit_percentage = ((final_amount - 100) / 100) * 100
            
            # Only proceed if we have a strictly positive profit
            if profit_percentage > 0.001:  # Using a small epsilon to account for floating point precision
                print(f"\nArbitrage opportunity found on {date.date()}!")
                print(f"Path: {' -> '.join(cycle)}")
                print(f"\nStarting with 100 {cycle[0]}")
                
                try:
                    # Calculate and show each conversion
                    current_amount = 100
                    for i in range(len(cycle)-1):
                        from_curr = cycle[i]
                        to_curr = cycle[i+1]
                        rate = np.exp(-G[from_curr][to_curr]['weight'])
                        current_amount *= rate
                        print(f"Convert {from_curr} to {to_curr}: {current_amount:.2f} {to_curr} (rate: {rate:.4f})")
                    
                    profit = current_amount - 100
                    profit_percentage = (profit / 100) * 100
                    print(f"\nTotal profit: {profit:.2f} {cycle[0]} ({profit_percentage:.2f}%)")
                    return True
                except Exception as e:
                    print(f"Error calculating arbitrage: {str(e)}")
                    continue
    
    return False

# Check each day in our data range
print("\nChecking for arbitrage opportunities...")
found_arbitrage = False

# Process each day until we find an arbitrage
for date in data.index:
    if check_arbitrage_for_day(data.loc[date], date):
        found_arbitrage = True
        break

if not found_arbitrage:
    print("\nNo arbitrage opportunities found in the last 2 years.")
