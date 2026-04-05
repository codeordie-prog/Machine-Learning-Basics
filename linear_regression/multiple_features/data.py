"""
Create the data.csv file with multiple features
features include - size, number of bedrooms, number of floors, age of house

"""

import csv

prices = [i * 100 for i in range(1, 101)]

def write_data(prices):
    
    with open("data.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["size_sqft", "num_bedrooms", "num_floors", "age_of_house_in_months", "price"])

        for i, price in enumerate(prices):
            writer.writerow([i+10, i+2, i+3, 12, price])
            
if __name__ == "__main__":
    write_data(prices)
    