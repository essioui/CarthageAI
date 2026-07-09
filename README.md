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
