import datetime as dt
import pandas as pd


def add_daytype(
    date: dt.datetime,
    holiday_list: tuple|None = None
) -> str:
    """
    Determines the type of day (e.g., Holiday, Saturday, Weekday) based on the provided date.

    Args:
        `date` (`datetime`): The date for which the day type is to be determined.
        `holiday_list` (`tuple`, optional): A tuple of strings representing holiday dates in the format `YYYY-MM-DD`.
            Defaults to a predefined list of holidays for the year 2024.

    Returns:
        `str`: The type of day: `'Holiday'`, `'Saturday'`, or `'Weekday'`.
    """
    if date.strftime('%Y-%m-%d') in holiday_list:
        return 'Holiday'
    elif date.weekday() == 5:
        return 'Saturday'
    elif date.weekday() < 5:
        return 'Weekday'
    else:
        return 'Holiday'