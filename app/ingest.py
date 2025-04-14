import pandas as pd
import glob
import json
import os
from pathlib import Path


def ingest_data(input_folders, output_file=None, top_revenue_countries_count=None):
    """
    Ingests JSON invoice files from multiple directories, extracts relevant fields,
    aggregates to monthly revenue per country for top 10 countries, and saves the result.

    Args:
        input_folders (list): List of paths to folders containing JSON files.
        output_file (str): Path to save the processed CSV file.

    Returns:
        pd.DataFrame: Processed DataFrame with monthly revenue data.
    """
    # Validate input folders
    valid_folders = [
        folder for folder in input_folders if os.path.exists(folder)]
    if not valid_folders:
        raise FileNotFoundError(
            f"None of the input folders exist: {input_folders}")

    if len(valid_folders) < len(input_folders):
        print(
            f"Warning: Some input folders do not exist: {[f for f in input_folders if f not in valid_folders]}")

    # Collect all JSON files across input folders
    json_files = []
    for folder in valid_folders:
        files = glob.glob(os.path.join(folder, "invoices-*.json"))
        json_files.extend(files)

    # Remove duplicates (if same file appears in multiple folders)
    json_files = list(set(json_files))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in folders: {valid_folders}")

    print(
        f"Found {len(json_files)} JSON files across {len(valid_folders)} folders: {valid_folders}")

    # Initialize list to store data
    data = []

    # Read each JSON file
    for file in json_files:
        try:
            with open(file, 'r') as f:
                # Load JSON content
                file_data = json.load(f)
                # Ensure file_data is a list of records
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    print(
                        f"Warning: File '{file}' is not a list of records. Skipping.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in '{file}': {e}. Skipping.")
        except Exception as e:
            print(f"Error reading '{file}': {e}. Skipping.")

    if not data:
        raise ValueError("No valid data loaded from JSON files.")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Check for required columns
    required_columns = ['country', 'price', 'year', 'month', 'day']
    missing_columns = [
        col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Optional column: times_viewed
    if 'times_viewed' not in df.columns:
        print("Warning: 'times_viewed' column not found. Proceeding without it.")
        df['times_viewed'] = 0  # Add dummy column to avoid conditional logic later

    # Select relevant columns
    df = df[['country', 'price', 'year', 'month', 'day', 'times_viewed']]

    # Data cleaning
    # Convert year, month, day, price to numeric, handle errors
    for col in ['year', 'month', 'day', 'price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['times_viewed'] = pd.to_numeric(
        df['times_viewed'], errors='coerce').fillna(0)

    # Drop rows with missing critical values
    df = df.dropna(subset=['country', 'price', 'year', 'month'])

    if df.empty:
        raise ValueError("No valid data after cleaning.")

    if top_revenue_countries_count is not None:
        print(f"Selecting top {top_revenue_countries_count} ...")
        # Identify top 10 countries by total revenue
        country_revenue = df.groupby(
            'country')['price'].sum().sort_values(ascending=False)
        top_10_countries = country_revenue.head(10).index
        print(f"Top 10 countries by revenue: {list(top_10_countries)}")

        # Filter to top 10 countries
        df = df[df['country'].isin(top_10_countries)]

    # Aggregate to monthly revenue per country
    df_monthly = df.groupby(['year', 'month', 'country']).agg({
        'price': 'sum',  # Total revenue
        'times_viewed': 'sum'  # Sum views
    }).reset_index()

    # Rename price to revenue for clarity
    df_monthly = df_monthly.rename(columns={'price': 'revenue'})

    # Create a date column for time-series analysis
    df_monthly['date'] = pd.to_datetime(
        df_monthly[['year', 'month']].assign(day=1))

    # Sort by date and country
    df_monthly = df_monthly.sort_values(['year', 'month', 'country'])

    # Save to CSV
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_monthly.to_csv(output_file, index=False)
        print(f"Processed data saved to '{output_file}'.")

    return df_monthly


if __name__ == "__main__":
    try:
        # Example: Load from multiple directories
        input_dirs = ["cs-train"]
        processed_df = ingest_data(
            input_folders=input_dirs, output_file="processed_revenue_data.csv", top_revenue_countries_count=10)
        print(f"Processed DataFrame shape: {processed_df.shape}")
        print(processed_df.head())
    except Exception as e:
        print(f"Error during ingestion: {e}")
