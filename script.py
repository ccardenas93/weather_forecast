import pandas as pd
from datetime import datetime

# Create a simple DataFrame with a timestamp
data = {
    'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
    'value': [42]  # arbitrary value
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('merged_data_export.csv', index=False)

print("CSV file updated with current timestamp.")