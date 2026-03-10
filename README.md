# ML for Time Series Incident/Anomaly Prediction

## Structure
```
project_folder/
├── data/            # all data (windowed, train, val, test)
├── results/         # model performance results
├── src/             # scripts for data processing and models
├── exp.ipynb        # visualize dataset, evluation results
├── README.md       
└── requirements.txt
```

Setup
```
conda create -n incident-predict
conda activate incident-predict
pip install -r requirements.txt
```

## Problem Formulation
**Goal:** Predict whether an incident occurs within next `H` time steps using previous `W` time steps.

**Problem Type:** Supervised machine learning (using labeled data), binary classification (0 = normal, 1 = abnormal) on multivariate time-series data (multiple metrics per time step).

**Data Representations**

Monitoring time-series data:
```
timestamp, measure(x)
```

Time steps formulation:
| Feature / Measures         | Description                                                     |
|:---------------- | :--------------------------------------------------------------- |
| `T1_x1`, `T1_x2` | Metric values at time step 1                                    |
| `T2_x1`, `T2_x2` | Metric values at time step 2                                    |
| ...              | ...                                                             |
| `TW_x1`, `TW_x2` | Metric values at time step W                                    |
| **y**            | Label indicating whether an incident occurs in the next H steps |

Sliding window formulation with multi-variates:
```
T1_x1, T1_x2, T2_x1, T2_x2, ..., TW_x1, TW_x2, y
```

This formulation allows the model to learn temporal dependencies in the metrics to predict upcoming incidents.

## Dataset Preparation

The dataset used in this work (under `./data/CINECA/raw_datasets/`) is sourced from a top-tier publication:  [Anomaly Detection and Anticipation in High Performance Computing Systems](https://ieeexplore.ieee.org/abstract/document/9439169). The dataset provides publicly available, well-structured monitoring metrics along with anomaly labels, and can be accessed at:  [Examon data from Marconi HPC system (snapshot)](https://zenodo.org/records/4537850).

The dataset is composed of two monitored periods January 2020, and May 2020. It aggregated in 5-minutes intervals. The features represent system monitoring metrics from HPC nodes, including environmental sensors, hardware status, CPU performance, memory usage, I/O activity, and network traffic. For each metric, both **average (avg)** and **variance (var)** values are provided to capture the level and variability of system behavior over time.

### Overview of the Selected Dataset

I selected a subset of the dataset based on the anomaly ratio, specifically `dataset_may_r162c05s02.csv`. This subset contains a sufficient number of anomaly labels for training, while allowing me to work within the limited time available to process the full dataset.

The data used for model input are as following:
| Split   |   #Total |   #Normal |   #Anomaly | %Anom   |
|:--------|--------:|------:|-------:|:-----------|
| train   |    5010 |  4053 |    957 | 19.10%     |
| val     |    1073 |   864 |    209 | 19.48%     |
| test    |    1075 |   862 |    213 | 19.81%     |

### Data Processing

1. Inspect each node and month, and select a sub-dataset that contains a sufficient number of both anomaly and normal samples. Based on this criterion, I selected `dataset_may_r162c05s02.csv`.
2. Transform the time-series data into windowed samples. Specifically, define the window size `W`, and prediction horizon `H`, and convert the time series into a sliding-window format. In this project, `W = 30min (6 steps)`, `H = 10min (2 steps)`.
3. Split the dataset into training, validation, and test sets. For simplicity, I randomly split the data using a 75% / 15% / 15% ratio. Ideally, time-series prediction should use **time-based splits** to avoid temporal leakage. However, I experimented with several sampling strategies, but they caused the model to predict either all normal or all abnormal samples. The reasons for this behavior will be discussed later in the *Limitations* section.



## Model Selection and Training

Three models are selected:

| Model               | Type              | Pros                                 |
| :------------------- | :----------------- | :--------------------------------------- |
| Logistic Regression | Linear            | Baseline performance                    |
| Random Forest       | Ensemble Tree     | Capture non-linear feature interactions |
| XGBoost             | Gradient Boosting | Strong high-performance model for tabular data |


Key considerations:
- **Feature Scaling.** Numeric telemetry metrics were standardized using feature scaling to ensure that features with larger magnitudes do not dominate the learning process.

- **Class Imbalance Handling.** Since anomalies occur less frequently than normal events, class weights were applied during training to mitigate class imbalance and prevent the model from biasing toward the majority class.

- **Threshold Adjustment.** Threshold Adjustment. The model outputs probabilities for anomaly prediction. By adjusting the probability threshold used to classify anomalies, the model’s sensitivity can be controlled. For example, a lower threshold increases recall (more anomalies are detected, including borderline cases) but may reduce precision, while a higher threshold increases precision (fewer false positives) but may reduce recall.

## Model Evaluation

### Evaluation Metrics

**Performance:** (1) Precision: the proportion of predicted anomalies that are actually true anomalies. (2) Recall: the proportion of true anomalies that are correctly detected. High recall indicates few missed anomalies. (3) F1-Score: the harmonic mean of precision and recall.
|                        | **Predicted Normal (0)** | **Predicted Anomaly (1)** |
| ---------------------- | ------------------------ | ------------------------- |
| **Actual Normal (0)**  | True Negative (TN)       | False Positive (FP)       |
| **Actual Anomaly (1)** | False Negative (FN)      | True Positive (TP)        |

**Latency:** (1) Train Time: the total time required to train the model on the training dataset. (2) Inference Time. The time needed to generate predictions for a single window or batch data.


### Result Analysis

| Model               |   Precision |   Recall |     F1 |   Train_time [s] |
|:--------------------|------------:|---------:|-------:|-----------------:|
| logistic_regression |      0.8048 |   0.9484 | 0.8707 |           0.4224 |
| random_forest       |      0.9952 |   0.9812 | 0.9882 |           0.3949 |
| xgboost             |      0.9859 |   0.9859 | 0.9859 |           2.8331 |

Key observations:
- Random Forest provides the best balance of accuracy and training speed, achieving an F1-score of 0.9882 with a training time of ~0.39 s for predicting anomalies from system telemetry.

- Logistic Regression serves as a useful baseline; however, its precision is lower (0.8048), meaning there are still some false positives.

- XGBoost achieves very high precision and recall (0.9859 each) and an F1-score of 0.9859, making it a strong alternative if additional tuning or handling of **more complex datasets** is desired, though its training time (~2.83 s) is longer.

## [TODO] Limitations and Improvements

1. Dataset split.
2. Feature engineering. Are the features useful for prediction?
3. Problem formulation. (1) Forcasting problem. (2) Multi-step prediction.
4. Models selection and training.
5. Model evaluation. Lead time for incident detection.

## [TODO] How to adapt it to cloud alert system?
1. Deployment.
2. Monitoring.
3. Retraining.


## Appendix

### A1. Task Division
- Day1: task exploration, problem formulation, dataset selection, data processing.
- Day2: model selection, traning, and evaluation.
- Day3: clean code, write README, task submission. 

### A2. Data and ML Pipeline

Data pipeline
```
1. merge_metrics        — merge per-metric CSVs into one file, resample to 5-min
2. add_anomaly          — label each timestamp with anomaly from combined_windows.json
3. sample               — select a time range and drop all-NaN columns
4. build_sliding_window — create (X, y) sliding-window dataset
5. split_dataset        — time-based train / val / test split
```

ML pipeline
```
1. data engineering
2. feature engineering
3. problem formulation
4. model selection
5. evaluation
6. deployment
7. monitoring
8. retraining
```


### A3. Other Issues
1. Firstly I found this data because it contains real data collected from CloudWatch: [Numenta Anomaly Benchmark (NAB)](https://www.kaggle.com/datasets/boltzmannbrain/nab/discussion/177967) and its [labeled data](https://github.com/numenta/NAB/tree/master/labels). I select the `./realAWSCloudWatch` time series metric data, because it is collected by CloudWatch and thus more related to the project description. The labeled anomaly time inteval is in `conbined_windows.json`. While I conduct the data processing pipeline, the model performance is really bad, either predict all class 0 or 1. I conduct several resampling but it didn't work at all.