import re
import pandas as pd


def convert_to_float(value):
    """
    Function to handle both $ symbol, thousands separator, and decimal separator
    Args:
        value: argument that contains the value to treat

    Returns:
        The value converted to a float or None
    """
    # Check if the value is a string
    if isinstance(value, str):
        # Remove the $ symbol
        value = value.replace('$', '')

        # Handle commas as thousands separator
        value = re.sub(r'(?<=\d),(?=\d{3}(\D|$))', '', value)

        # Replace commas as decimal separator with a dot
        value = value.replace(',', '.')

        # Replace %
        value = value.replace("%", "")

        # Convert to float (if not empty)
        return float(value) if value else None

    # If the value is already a float, return it as is
    elif isinstance(value, float):
        return value

    # For other types (e.g., None), return None
    else:
        return None

def number_of_days_passed(series: pd.Series, day: str) -> pd.Series:
    # Convert the series to datetime
    series_datetime = pd.to_datetime(series)

    # Find the latest date in the series
    latest_date = pd.to_datetime(day)

    # Calculate the number of days passed for each value
    days_passed = (latest_date - series_datetime).dt.days

    return days_passed