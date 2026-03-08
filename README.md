# ML for Time-series Incident Prediction

## Task Description

Task #1
Implement a model that predicts whether an incident will occur within the next H time steps based on the previous W steps of one or more time-series metrics. Use a sliding-window formulation and train the model using any standard machine-learning framework.

The applicant may use any suitable public dataset or generate a synthetic time series with labeled incident intervals (e.g. anomalies or threshold breaches). The emphasis is on correct problem formulation, model selection, training, and evaluation rather than dataset complexity or model size.

The solution should include a clear description of the modeling choices, the evaluation setup (including alert thresholds and metrics), and an analysis of the results. During follow-up, the applicant should be able to explain the design decisions, discuss limitations, and outline how the approach could be adapted to a real alerting system.
Preferably, solutions should be provided as a link to a public GitHub repository.



## Problem Formulation
Goal: Predict whether an incident occurs within next H time steps using previous W time steps.

Problem Type: Supervised Machine Learning Classification Task on or Multi-variate Time Series Data

## Dataset Preparasion

### Overview of the Dataset
Data Source: [Numenta Anomaly Benchmark (NAB)](https://www.kaggle.com/datasets/boltzmannbrain/nab/discussion/177967) and its [labeled data](https://github.com/numenta/NAB/tree/master/labels).
For convinience, we only select the `./realAWSCloudWatch` time series metric data. The label anomaly time inteval is in `conbined_windows.json`.

(NOTE: To simplify, in this project, we see `anomaly` and `incident` as same concepts.)


### Data Processing

1. Merge metrics. seperate metric CSV: timestamp, value   -> merged metric csv: timestamp, value1, value2, ...,. Input: `./data/realAWSCloudwatch`.
2. Add anomaly labels based on `conbined_windows.json`. Output: `merged_metrics_with_anomaly.csv`.
3. Select dense rows and drop NaN columns. Output: `merged_metrics_with_anomaly_samples.csv`.
4. Define the value W (window size) and H, then transform time-series to slide window format, for later formulating training samples. Output: `merged_slide_window.csv`.


## Model Selection

- Regression
- XGboost
- LSTM

## Model Training

## Results Evaluation

### Metrics
Precision
Recall
F1-Score

## Limitations and Improvements

## Adaption Discussion



## Appendix

### A1. Cloud Incident Datasets
