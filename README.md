# Experiment for SolarBoost

This is the code for SolarBoost, a boosting method for distributed photovoltaic power forecasting.

## Quick Start Guide

To run experiments, use `exp.py` with one of the following options:

1. `table2` - Generates results for Table 2 (AR grid analysis)
2. `table3` - Generates results for Table 3 (Aggregate output analysis for AR and Kalman datasets) 
3. `table4` - Generates results for Table 4 (City A dataset analysis)
4. `figure9` - Generates capacity plots for Figure 9

## Data and results
If a model with the specified parameters does not exist, it will be automatically trained and saved to the `./models` directory.

Real data for city A is saved as `./data/0.2.npz`.
Figures and Tables are saved in `./figures` and `./table` folders.

## Parameters
Parameters are set in `ar1.py`, `kalman.py` and `city_a.py`, respectively.


