## Dataset Overview

This project is based on the **Individual Household Electric Power Consumption** dataset, used for analyzing and forecasting household electricity consumption., originally published by the **UCI Machine Learning Repository** and also available on Kaggle.

The dataset contains minute-level measurements of household electricity consumption collected over **47 months**, from **December 2006 to November 2010**.

| Property           | Value     |
| ------------------ | --------- |
| Sampling Frequency | 1 minute  |
| Time Period        | 47 months |
| Observations       | 2,075,259 |
| Features           | 9         |

### Dataset Source

- UCI Machine Learning Repository:
  https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

- Kaggle:
  https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set

## Features Description

The dataset consists of nine features describing household electricity consumption measured every minute.

| Feature                   | Unit           | Description                                                                                                                                                          |
| ------------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Date**                  | -              | Date of the measurement.                                                                                                                                             |
| **Time**                  | -              | Time of the measurement.                                                                                                                                             |
| **Global_active_power**   | kilowatt (kW)  | Total active power consumed by the household. Active power represents the actual power used to perform work by electrical devices.                                   |
| **Global_reactive_power** | kilowatt (kW)  | Reactive power consumed by the household. It represents energy exchanged between the electrical system and inductive loads without being converted into useful work. |
| **Voltage**               | Volt (V)       | Average voltage measured during the one-minute interval.                                                                                                             |
| **Global_intensity**      | Ampere (A)     | Average current intensity measured during the one-minute interval.                                                                                                   |
| **Sub_metering_1**        | watt-hour (Wh) | Energy consumption of the kitchen, mainly dishwasher, oven, and microwave.                                                                                           |
| **Sub_metering_2**        | watt-hour (Wh) | Energy consumption of the laundry room, mainly washing machine, tumble dryer, refrigerator, and light.                                                               |
| **Sub_metering_3**        | watt-hour (Wh) | Energy consumption of the electric water heater and air conditioner.                                                                                                 |

## Exploratory Data Analysis (EDA)

Before building machine learning models, an exploratory analysis is performed to understand the structure, quality, and patterns of the dataset.

The main objectives of EDA are:

- Verify the completeness of the time series.
- Detect missing values and inconsistencies.
- Understand consumption patterns over time.
- Analyze relationships between electrical measurements.
- Identify important features for modeling.

### 1. Time Series Completeness Check

**Question**

Does the dataset contain one measurement for every minute between the first and last timestamp?

**Analysis**

The first step is to verify that the time series is complete by comparing the number of observations in the dataset with the expected number of one-minute intervals between the first and last timestamps.

The expected number of minutes is calculated from the minimum and maximum timestamps, then compared with the total number of rows in the dataset.

**Result**

The dataset contains **2,075,259** observations, which exactly matches the expected number of one-minute intervals between the first and last timestamps. This confirms that **no timestamps are missing** and the time series is complete.

Although every timestamp is present, approximately **1.25%** of the records contain missing measurement values (`NaN`) in one or more electrical variables. Therefore, the dataset has **missing values but no missing time intervals**.

| Metric           |               Value |
| ---------------- | ------------------: |
| Start Timestamp  | 2006-12-16 17:24:00 |
| End Timestamp    | 2010-11-26 21:02:00 |
| Expected Minutes |           2,075,259 |
| Actual Rows      |           2,075,259 |
| Missing Minutes  |                   0 |

### 2. Preparing Data for Time Series Analysis

After verifying that the time series is complete, the dataset is prepared for time series analysis.

The following preprocessing steps are performed:

- Combine the `Date` and `Time` columns into a single `Datetime` column.
- Convert the `Datetime` column to a proper datetime format.
- Set `Datetime` as the index of the dataset.
- Remove the original `Date` and `Time` columns since they are no longer required.

Using `Datetime` as the index allows efficient time-based operations such as resampling, rolling statistics, and analyzing consumption patterns over different time periods.

The final dataset structure is:

| Index    | Features                |
| -------- | ----------------------- |
| Datetime | Electrical measurements |

Example:

```python
df["Datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True
)

df = df.set_index("Datetime")

df = df.drop(columns=["Date", "Time"])

df = df.sort_index()
```

---

### 3. Dataset Overview and Data Quality Assessment

After preparing the dataset for time series analysis, the structure and quality of the data are examined.

The following checks are performed:

- Dataset dimensions (number of rows and columns).
- Data types of each feature.
- Missing values detection.
- Missing values percentage.
- Duplicate rows detection.

---

#### Dataset Shape

The dataset contains **2,075,259 observations** and **10 columns** before removing the original `Date` and `Time` columns.

| Metric            |     Value |
| ----------------- | --------: |
| Number of Rows    | 2,075,259 |
| Number of Columns |        10 |

---

#### Data Types and Dataset Information

The dataset contains:

- 2 datetime columns:
  - `Datetime`
- 2 categorical/text columns:
  - `Date`
  - `Time`
- 7 numerical measurement features:
  - `Global_active_power`
  - `Global_reactive_power`
  - `Voltage`
  - `Global_intensity`
  - `Sub_metering_1`
  - `Sub_metering_2`
  - `Sub_metering_3`

| Data Type  | Number of Columns |
| ---------- | ----------------: |
| datetime64 |                 1 |
| float64    |                 7 |
| object     |                 2 |

The dataset requires approximately **158.3 MB** of memory.

---

#### Missing Values Analysis

Missing values are checked for each feature.

The results show that missing values exist only in electrical measurements, while the timestamp information (`Date`, `Time`, and `Datetime`) is complete.

| Feature               | Missing Values | Percentage |
| --------------------- | -------------: | ---------: |
| Global_active_power   |         25,979 |      1.25% |
| Global_reactive_power |         25,979 |      1.25% |
| Voltage               |         25,979 |      1.25% |
| Global_intensity      |         25,979 |      1.25% |
| Sub_metering_1        |         25,979 |      1.25% |
| Sub_metering_2        |         25,979 |      1.25% |
| Sub_metering_3        |         25,979 |      1.25% |
| Date                  |              0 |         0% |
| Time                  |              0 |         0% |
| Datetime              |              0 |         0% |

Approximately **1.25% of the measurements contain missing values**. These missing values will be handled during the data preprocessing stage.

---

#### Duplicate Rows Check

Duplicate observations are checked to ensure data consistency.

| Metric         | Value |
| -------------- | ----: |
| Duplicate Rows |     0 |

The dataset does not contain duplicated rows.

---

### 4. Missing Values Treatment

The dataset contains approximately **1.25%** missing values in the electrical measurement variables.

Since the data represents a continuous time series with measurements recorded every minute and **no timestamps are missing**, interpolation is an appropriate method for estimating the missing observations.

Time-based interpolation was selected because it preserves the temporal continuity of the data by estimating missing values from neighboring observations rather than removing records.

---

### 5. Outlier Detection

Outliers were analyzed using the **Interquartile Range (IQR) method** on all numerical features.

For each variable, the following values were calculated:

- First quartile (Q1)
- Third quartile (Q3)
- Interquartile Range (IQR)
- Lower and upper bounds
- Number and percentage of detected outliers

| Feature               |    IQR | Lower Bound | Upper Bound | Number of Outliers | Percentage |
| --------------------- | -----: | ----------: | ----------: | -----------------: | ---------: |
| Global_active_power   |  1.218 |      -1.517 |       3.355 |             95,238 |      4.59% |
| Global_reactive_power |  0.146 |      -0.171 |       0.413 |             40,478 |      1.95% |
| Voltage               |  3.880 |     233.170 |     248.690 |             52,195 |      2.52% |
| Global_intensity      |  5.000 |      -6.100 |      13.900 |            100,979 |      4.87% |
| Sub_metering_1        |  0.000 |       0.000 |       0.000 |            172,247 |      8.30% |
| Sub_metering_2        |  1.000 |      -1.500 |       2.500 |             77,166 |      3.72% |
| Sub_metering_3        | 17.000 |     -25.500 |      42.500 |                  0 |      0.00% |

---

### 6. Outlier Treatment

The IQR analysis identified outliers in several electrical measurement variables. The highest proportion was observed in `Sub_metering_1` (8.30%), followed by `Global_intensity` (4.87%) and `Global_active_power` (4.59%).

However, these observations were carefully examined before deciding whether they should be removed.

In household electricity consumption data, high values usually correspond to periods when multiple electrical appliances operate simultaneously. Therefore, these observations represent **real consumption behavior** rather than measurement errors.

For the `Sub_metering` variables, the detected outliers are mainly a consequence of their highly skewed distributions. Since these variables contain a large number of zero values, the IQR method classifies many legitimate non-zero observations as outliers.

Removing these values would eliminate important consumption peaks and distort the temporal dynamics of the time series, which could negatively affect forecasting models.

Therefore, **no outliers were removed**, and all observations were retained for subsequent preprocessing and modeling.

---

## 7. Daily Energy Consumption

The original dataset records **active power (`Global_active_power`)** in kilowatts (kW) every minute. However, the forecasting task focuses on predicting **daily household energy consumption**.

To obtain daily energy usage, the power measurements are converted into energy using the physical relationship:

\[
Energy\;(kWh)=Power\;(kW)\times Time\;(h)
\]

Since each observation represents one minute,

\[
Energy\_{minute}(kWh)=Global_active_power\times\frac{1}{60}
\]

where:

- `Global_active_power` is the active power measured in kilowatts.
- \(\frac{1}{60}\) converts one minute into hours.

Finally, the daily energy consumption is calculated by summing the one-minute energy values for each day.

This aggregation transforms the original minute-level dataset into a daily time series that is more suitable for forecasting while preserving the total daily electricity consumption.

---

## 8. Daily Consumption Analysis

The daily energy consumption time series was analyzed to better understand its temporal behavior before developing forecasting models.

The analysis reveals several important characteristics:

- Electricity consumption exhibits a clear seasonal pattern, with recurring increases and decreases throughout the observation period.
- Daily consumption fluctuates considerably, reflecting variations in household electricity usage.
- No strong long-term upward or downward trend is observed over the 47-month period, suggesting relatively stable overall consumption.
- Several days show unusually high electricity usage, corresponding to periods of increased household demand rather than measurement errors.
- These observations indicate that temporal features (such as month, season, and day of the week) and external variables (such as temperature) are likely to improve forecasting performance.

Overall, the dataset presents the typical characteristics of an energy consumption time series, making it well suited for time series forecasting using both statistical and machine learning approaches.

---

# Data Preprocessing and Feature Engineering

After completing the exploratory data analysis, the raw dataset is transformed into a clean and structured dataset suitable for machine learning and deep learning models.

The preprocessing pipeline aims to:

- Handle missing observations while preserving temporal continuity.
- Generate informative temporal features.
- Convert minute-level power measurements into daily energy consumption.
- Integrate external weather information.
- Produce a final processed dataset ready for model training.

The complete preprocessing workflow is summarized below.

| Step                         | Description                                                        |
| ---------------------------- | ------------------------------------------------------------------ |
| Missing Value Treatment      | Fill missing measurements using time-based interpolation           |
| Temporal Feature Engineering | Extract calendar-related features from the datetime index          |
| Daily Energy Aggregation     | Convert minute-level power into daily household energy consumption |
| Feature Selection            | Remove unnecessary electrical variables                            |
| Weather Data Integration     | Add daily weather observations from Meteostat                      |
| Export Processed Dataset     | Save the final dataset for machine learning                        |

---

## 1. Missing Value Treatment

Approximately **1.25%** of the electrical measurements contain missing values. Since the dataset represents a continuous one-minute time series with no missing timestamps, missing observations are estimated using **time-based interpolation**.

Unlike deleting observations, interpolation preserves the continuity of the time series and minimizes information loss.

```python
df = df.interpolate(method="time")
```

---

## 2. Temporal Feature Engineering

Electricity consumption usually depends on calendar-related patterns such as weekdays, weekends, months, and seasons.

To help the forecasting models learn these temporal behaviors, several features are extracted from the `Datetime` index.

| Feature     | Description                             |
| ----------- | --------------------------------------- |
| `Day`       | Day of the month                        |
| `Month`     | Month of the year                       |
| `Year`      | Calendar year                           |
| `DayOfWeek` | Day of the week (Monday = 0)            |
| `IsWeekend` | Weekend indicator (Saturday and Sunday) |

```python
df["Day"] = df.index.day
df["Month"] = df.index.month
df["Year"] = df.index.year
df["DayOfWeek"] = df.index.dayofweek
df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
```

These engineered features allow machine learning models to capture weekly and seasonal consumption patterns.

---

## 3. Daily Energy Consumption Aggregation

The original dataset records **active power (`Global_active_power`)** every minute in **kilowatts (kW)**.

Since the forecasting task focuses on **daily household energy consumption**, the minute-level measurements are converted into energy using:

\[
Energy*{minute}(kWh)=Power*{minute}(kW)\times\frac{1}{60}
\]

The total daily energy consumption is then obtained by summing all one-minute energy values within each day.

```python
df["Daily_Energy_KWh"] = df["Global_active_power"] / 60

df["Daily_Energy_KWh"] = (
    df["Daily_Energy_KWh"]
    .resample("D")
    .sum()
)
```

The resulting variable, **Daily_Energy_KWh**, becomes the prediction target for the forecasting models.

---

## 4. Feature Selection

After creating the target variable, several original electrical measurements are removed because they are no longer required.

The removed features are:

- `Global_active_power`
- `Global_reactive_power`
- `Global_intensity`
- `Voltage`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`

Removing redundant variables simplifies the dataset while keeping the most informative features for forecasting.

---

## 5. Weather Data Integration

Household electricity consumption is strongly influenced by weather conditions.

To capture this relationship, daily weather observations are retrieved from the **Meteostat** database using the geographic coordinates of **Sceaux, France**, where the household is located.

The following weather variables are incorporated into the dataset.

| Feature | Description                    |
| ------- | ------------------------------ |
| `tavg`  | Average daily temperature (°C) |
| `tmin`  | Minimum daily temperature (°C) |
| `tmax`  | Maximum daily temperature (°C) |

```python
sceaux = Point(48.7763, 2.2903)

weather = Daily(sceaux, start_date, end_date).fetch()

weather = weather[
    ["tavg", "tmin", "tmax"]
]

df = df.merge(
    weather,
    left_index=True,
    right_index=True,
    how="left"
)
```

Including weather variables enables forecasting models to better explain variations in household electricity demand caused by environmental conditions.

---

## 6. Final Processed Dataset

The processed dataset is exported for subsequent machine learning experiments.

```python
df.to_csv(
    "../data/processed/household_power_consumption_processed.csv",
    index=True
)
```

### Final Dataset

The resulting dataset contains one observation per day and includes both temporal and weather-related features.

| Feature            | Type      | Description                        |
| ------------------ | --------- | ---------------------------------- |
| `Daily_Energy_KWh` | Target    | Daily household energy consumption |
| `Day`              | Numerical | Day of the month                   |
| `Month`            | Numerical | Month of the year                  |
| `Year`             | Numerical | Calendar year                      |
| `DayOfWeek`        | Numerical | Day of the week                    |
| `IsWeekend`        | Binary    | Weekend indicator                  |
| `tavg`             | Numerical | Average daily temperature          |
| `tmin`             | Numerical | Minimum daily temperature          |
| `tmax`             | Numerical | Maximum daily temperature          |

The final processed dataset is clean, complete, and enriched with temporal and weather-related information. It serves as the input for the machine learning and deep learning models developed in the next stage of this project.
