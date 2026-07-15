import pandas as pd
from meteostat import Point, Daily

# Load dataset
df = pd.read_csv(
    '../data/raw/household_power_consumption.txt',
    sep=';',
    na_values='?'
)

# Create new column 'Datetime' by combining 'Date' and 'Time' columns
df['Datetime'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'],
    dayfirst=True
)

# Set 'Datetime' as the index and drop the original 'Date' and 'Time' columns
df = (
    df.drop(columns=['Date', 'Time'])
    .set_index('Datetime')
)

# Interpolate missing values using time-based interpolation
df = df.interpolate(method='time')
 
# features engineering
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Year'] = df.index.year
df['DayOfWeek'] = df.index.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Calculate daily energy consumption in kWh
df['Daily_Energy_KWh'] = df['Global_active_power'] / 60
df['Daily_Energy_KWh'] = df['Daily_Energy_KWh'].resample('D').sum()

# Drop unnecessary columns
df = df.drop(columns=['Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'Global_active_power', 'Global_reactive_power', 'Voltage'])

# Print the start and end dates of the dataset
start_date = df.index.min()
end_date = df.index.max()
print(f"Start date: {start_date}, End date: {end_date}")



# Coordinates of Sceaux, France
sceaux = Point(48.7763, 2.2903)

# Download daily weather data
weather = Daily(sceaux, start_date, end_date)
weather = weather.fetch()

# Select only the relevant columns
weather = weather[['tavg', 'tmin', 'tmax']]

# Merge the weather data with the original dataset
df = df.merge(
    weather,
    left_index=True,
    right_index=True,
    how='left'
)

# save file csv
df.to_csv('../data/processed/household_power_consumption_processed.csv', index=True)
