import pandas as pd


def to_utc_naive(timestamp):
    timestamp = pd.Timestamp(timestamp)
    if timestamp.tzinfo is None:
        return timestamp
    return timestamp.tz_convert(None)


def parse_timestamp_column(data, column):
    if column in data.columns:
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce").dt.tz_convert(None)
    return data
