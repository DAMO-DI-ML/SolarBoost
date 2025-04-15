# SolarBoost: Distributed Photovoltaic Power Forecasting

<!-- [![Made by DAMO Academy](https://img.shields.io/badge/Made%20by-DAMO%20Academy-blue)](https://damo.alibaba.com) -->

SolarBoost is an advanced boosting method for distributed photovoltaic (PV) power forecasting, developed by DAMO Academy, Alibaba Group. This repository contains the implementation and experimental code for the SolarBoost algorithm.

## Features

- Accurate forecasting for distributed photovoltaic power systems
- Boosting-based methodology for improved prediction accuracy
- Support for multiple datasets including AR, Kalman, and real-world city data
- Comprehensive experimental analysis and benchmarking

## Quick Start Guide

To reproduce our experimental results, run `exp.py` with one of the following options:

1. `python exp.py table2` - Reproduces Table 2 (AR grid analysis)
2. `python exp.py table3` - Reproduces Table 3 (Aggregate output analysis for AR and Kalman datasets)
3. `python exp.py table4` - Reproduces Table 4 (City A dataset analysis)
4. `python exp.py figure9` - Generates capacity plots for Figure 9

## Project Structure

- `./data/` - Contains the datasets
  - `0.2.npz` - Real-world data from City A
- `./models/` - Stores trained models
- `./figures/` - Output directory for generated figures
- `./tables/` - Output directory for result tables

## Model Training

Models are automatically trained if they don't exist for the specified parameters. Configuration parameters can be found in:
- `ar1.py` - AR model parameters
- `kalman.py` - Kalman filter parameters
- `city_a.py` - City A dataset parameters

<!-- ## Citation

If you use SolarBoost in your research, please cite our paper: -->


