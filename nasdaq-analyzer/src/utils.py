# Utility functions for the Nasdaq Analyzer application

def load_data(file_path):
    """Load data from a CSV file."""
    import pandas as pd
    try:
        return pd.read_csv(file_path, parse_dates=True)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def save_data(df, file_path):
    """Save DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")

def format_date(date):
    """Format date for display."""
    return date.strftime("%Y-%m-%d")

def calculate_percentage_change(current, previous):
    """Calculate percentage change between two values."""
    if previous == 0:
        return None
    return (current - previous) / previous * 100

def get_unique_values(column):
    """Get unique values from a DataFrame column."""
    return column.unique().tolist()