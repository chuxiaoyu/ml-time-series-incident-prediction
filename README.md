# ML for Time Series Incident Prediction

## Task Description

Task #1
Implement a model that predicts whether an incident will occur within the next H time steps based on the previous W steps of one or more time-series metrics. Use a sliding-window formulation and train the model using any standard machine-learning framework.

The applicant may use any suitable public dataset or generate a synthetic time series with labeled incident intervals (e.g. anomalies or threshold breaches). The emphasis is on correct problem formulation, model selection, training, and evaluation rather than dataset complexity or model size.

The solution should include a clear description of the modeling choices, the evaluation setup (including alert thresholds and metrics), and an analysis of the results. During follow-up, the applicant should be able to explain the design decisions, discuss limitations, and outline how the approach could be adapted to a real alerting system.
Preferably, solutions should be provided as a link to a public GitHub repository.

## Structure

## Problem Formulation
Goal: Predict whether an incident occurs within next H time steps using previous W time steps.

Problem Type: Supervised Machine Learning Binary-Classification Task on Multi-Variate Time Series Data

## Dataset Preparasion

### Overview of the Dataset
Data Source: [Numenta Anomaly Benchmark (NAB)](https://www.kaggle.com/datasets/boltzmannbrain/nab/discussion/177967) and its [labeled data](https://github.com/numenta/NAB/tree/master/labels).
For convinience, we only select the `./realAWSCloudWatch` time series metric data, because it is collected by CloudWatch and thus more related to the project description. The labeled anomaly time inteval is in `conbined_windows.json`.

(NOTE: In this project, we see *anomaly* and *incident* as same concepts.)


### Data Processing

1. Merge metrics. seperate metric CSV: timestamp, value   -> merged metric csv: timestamp, value1, value2, ...,. Input: `./data/realAWSCloudwatch`.
2. Add anomaly labels based on `conbined_windows.json`. Output: `merged_metrics_with_anomaly.csv`.
3. Select dense rows and drop NaN columns. Output: `merged_metrics_with_anomaly_samples.csv`.
4. Define the value W (window size) and H, then transform time-series to slide window format, for later formulating training samples. Output: `merged_slide_window.csv`.


## Model Selection

- LogisticRegression
- RandomForest
- XGboost

## Model Training

## Model Evaluation

### Metrics
- Performance: Precision, Recall, F1-Score.
- Latency: Train Time, Inference Time.
- Resource Usage.

## Limitations and Improvements

1. Data engineering.
2. Feature engineering. Are the features useful for prediction?
3. Problem formulation. Multi-step prediction.
4. Models Training.
5. Model Evaluation. Lead time for incident detection.

## How to adapt it to cloud alert system?
1. Deployment.
2. Monitoring.
3. Retraining.


## Appendix

### A1. Time Budget
- Day1: task exploration, problem formulation, dataset selection, data processing.
- Day2: model selection, traning, and evaluation.
- Day3: clean code, write README, task submission. 

## A2. ML Pipeline

```
data engineering
feature engineering
problem formulation
model selection
evaluation
deployment
monitoring
retraining
```


## A3. Key Issues
1. Data Sampling.
2. Feature Engineering.
3. Accuracy.