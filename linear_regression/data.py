"""
Writes a data.csv file with the following format:
House size (1000 sq ft) | Price ($)
------------------------|------------
1                       | 300
2                       | 500

"""
import csv

# Initilize prices 
prices = [i * 100 for i in range(1, 101)]

def write_data(prices) -> None:
    """
    Writes a data.csv file with the given prices
    
    Args:
        prices (list): List of prices
    
    Returns:
        None
    """
    
    with open("data.csv", 'w') as f:
        # get a writer
        writer = csv.writer(f)

        # write the header
        writer.writerow(["House size (1000 sq ft)", "Price ($)"])
        
        # write the data
        for i, price in enumerate(prices):
            writer.writerow([i + 1, price])
        
if __name__ == "__main__":
    write_data(prices)
