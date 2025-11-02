# data_acquisition.py
import pandas as pd
import requests
import io
from datetime import datetime, timedelta
import time

# Define the base URL for the USGS API
base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"

# Define our parameters. We'll change the time range for each request.
params = {
    'starttime': None,  # We will set these in the loop
    'endtime': None,
    'minmagnitude': 4.5,
    'orderby': 'time-asc',  # Crucial: Order by time ascending to chunk correctly
    'limit': 20000         # Explicitly set the limit to the maximum allowed
}

# Define the overall date range for our entire dataset
start_year = 2010  # We start in 2010 to manage size. We can go back further later.
end_year = 2024
current_start = datetime(start_year, 1, 1)  # Jan 1, 2010
current_end = datetime(start_year, 12, 31)  # Dec 31, 2010

# Create an empty list to store each chunk of data we download
all_chunks = []

# Loop through each year from start_year to end_year
print("Starting data download... This may take a few minutes.")
for year in range(start_year, end_year + 1):
    print(f"Downloading data for year: {year}")
    
    # Set the parameters for this specific year
    params['starttime'] = f"{year}-01-01"
    params['endtime'] = f"{year}-12-31"
    
    # Make the request to the USGS API
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # This will raise an error for bad status codes (4xx or 5xx)
        
        # Check if the response is empty (no earthquakes that year)
        if response.text.strip() == '':
            print(f"  No data for {year}.")
            continue
            
        # Read the CSV data directly from the response text into a DataFrame
        df_chunk = pd.read_csv(io.StringIO(response.text))
        
        # Print the number of events found for feedback
        print(f"  Downloaded {len(df_chunk)} events.")
        
        # Append the chunk to our list
        all_chunks.append(df_chunk)
        
    except requests.exceptions.HTTPError as err:
        # If we hit a 400 error (like the limit), we need to break the year into smaller pieces
        print(f"  HTTP Error for {year}: {err}. Breaking into smaller chunks...")
        # We will implement monthly chunks here if needed.
        # For now, we note the error and move on.
    except Exception as e:
        print(f"  An unexpected error occurred for {year}: {e}")
    
    # Be a good citizen: pause briefly between requests to not overload the server
    time.sleep(1)  # 1 second pause

# After the loop, combine all the chunks into one master DataFrame
if all_chunks: # Check if the list is not empty
    print("Combining all chunks into master DataFrame...")
    df_master = pd.concat(all_chunks, ignore_index=True)
    
    # Save the master DataFrame to a CSV file for future use
    output_file = f'usgs_earthquakes_{start_year}_{end_year}.csv'
    df_master.to_csv(output_file, index=False)
    print(f"Done! Master dataset saved to '{output_file}'.")
    print(f"Total events downloaded: {len(df_master)}")
    
    # Let's do a quick check of our data
    print("\n=== First Look at Master Data ===")
    print(df_master.info())
    
else:
    print("No data was downloaded.")