# References for using the yfiance libary/API for downloading data
# https://medium.com/@euricopaes/extracting-data-from-yahoo-finance-with-yfinance-96798253d8ca
# https://blog.quantinsti.com/download-forex-price-data-yfinance-library-python/

#Reference for the Bellman-Ford algorithm
# https://www.geeksforgeeks.org/dsa/bellman-ford-algorithm-dp-23/

# Top currencies used
# https://www.xs.com/en/blog/strongest-currencies-in-the-world/


import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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

# Download exchange rate data
print(f"Downloading data from {start_date.date()} to {end_date.date()}")
raw_data = yf.download(pairs, start=start_date, end=end_date)

# Get the Close prices and drop any columns with all NaN values
data = raw_data['Close'].dropna(axis=1, how='all')

print("\nAvailable currency pairs:")
print(data.columns.tolist())
print("\nSample exchange rates:")
print(data.iloc[-1].head())

def bellman_ford_arbitrage(graph, source):
    # Initialize distances and predecessors
    distances = {node: float('inf') for node in graph.nodes()}
    predecessors = {node: None for node in graph.nodes()}
    distances[source] = 0
    
    # Relax edges repeatedly
    for _ in range(len(graph.nodes()) - 1):
        for u, v, data in graph.edges(data=True):
            if distances[u] + data['weight'] < distances[v]:
                distances[v] = distances[u] + data['weight']
                predecessors[v] = u
    
    # Check for negative cycles
    for u, v, data in graph.edges(data=True):
        if distances[u] + data['weight'] < distances[v]:
            # Found a negative cycle, reconstruct the path
            cycle = []
            visited = set()
            current = v
            
            while current not in visited:
                visited.add(current)
                cycle.append(current)
                current = predecessors[current]
            
            # Get the cycle starting from the first occurrence of current
            start_idx = cycle.index(current)
            cycle = cycle[start_idx:]
            cycle.append(current)  # Complete the cycle
            
            return cycle
    
    return None

def check_arbitrage_for_day(rates, date):
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges with raw exchange rates
    for pair, rate in rates.items():
        if pd.isna(rate):
            continue  # Skip missing rates

        base = pair[:3]
        quote = pair[3:6]

        # In order for Bellman-Ford to work, we need to negate the log of the rate
        # This is because Bellman-Ford is a shortest path algorithm, and we want to find the most
        # profitable cycle, which is equivalent to finding the shortest path after we take 
        # the inverse log of the actual exchange rate
        try:
            G.add_edge(base, quote, weight=-np.log(rate))
            G.add_edge(quote, base, weight=np.log(rate))
        except Exception as e:
            print(f"Skipping {pair} due to error: {e}")

    print("\nGraph edges and weights (first 10):")
    edge_count = 0
    for u, v, data in G.edges(data=True):
        if edge_count < 10:  # Only show first 10 edges
            print(f"{u} -> {v}: {data['weight']}")
            edge_count += 1
        else:
            break
    
    # Check for arbitrage opportunities
    for source in G.nodes():
        cycle = bellman_ford_arbitrage(G, source)
        if cycle:
            # Verify that all edges in the cycle exist
            valid_cycle = True
            for i in range(len(cycle)-1):
                if not G.has_edge(cycle[i], cycle[i+1]):
                    valid_cycle = False
                    break
            # Check the final edge back to start
            if not G.has_edge(cycle[-1], cycle[0]):
                valid_cycle = False
                
            if not valid_cycle:
                continue
                
            print(f"\nArbitrage opportunity found on {date.date()}!")
            print(f"Path: {' -> '.join(cycle)}")
            
            # Calculate the total profit
            current_amount = 100  # Start with 100 units of the initial currency
            print(f"\nStarting with {current_amount} {cycle[0]}")
            
            try:
                # Calculate the path and show each conversion
                for i in range(len(cycle)-1):
                    from_curr = cycle[i]
                    to_curr = cycle[i+1]
                    edge_data = G.get_edge_data(from_curr, to_curr)
                    if edge_data is None:
                        raise ValueError(f"No exchange rate found for {from_curr} to {to_curr}")
                    rate = np.exp(-edge_data['weight'])  # Convert back from log space
                    current_amount *= rate
                    print(f"Convert {from_curr} to {to_curr}: {current_amount:.2f} {to_curr} (rate: {rate:.4f})")
                
                # Final conversion back to starting currency
                from_curr = cycle[-1]
                to_curr = cycle[0]
                edge_data = G.get_edge_data(from_curr, to_curr)
                if edge_data is None:
                    raise ValueError(f"No exchange rate found for {from_curr} to {to_curr}")
                rate = np.exp(-edge_data['weight'])
                final_amount = current_amount * rate
                print(f"Convert {from_curr} back to {to_curr}: {final_amount:.2f} {to_curr} (rate: {rate:.4f})")
                
                profit = final_amount - 100
                profit_percentage = (profit / 100) * 100
                print(f"\nTotal profit: {profit:.2f} {cycle[0]} ({profit_percentage:.2f}%)")
                return True
            except Exception as e:
                print(f"Error calculating arbitrage: {str(e)}")
                continue
    
    return False

# Check each day in our data range
print("\nChecking for arbitrage opportunities on each day...")
found_arbitrage = False

# Process each day until we find an arbitrage
for date in data.index:
    print(f"\nChecking {date.date()}...")
    if check_arbitrage_for_day(data.loc[date], date):
        found_arbitrage = True
        print(f"\nStopping search - arbitrage opportunity found on {date.date()}")
        break

if not found_arbitrage:
    print("\nNo arbitrage opportunities found in the last 2 years.")