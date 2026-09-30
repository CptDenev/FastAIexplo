# Predictive maintenance: MLP vs Random Forest

## Problem
Predict machine failures from sensor data, where missing a failure costs far more than a false alarm.

## Data
[AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI), 10,000 rows, 339 failures (about 1 failure for 28 normal cases). Public, synthetic dataset modeled on real industrial conditions.

## Approach
Same split and preprocessing for both models:
- MLP in PyTorch
- Random Forest in scikit-learn

## Results

| Model | Global accuracy | Failure recall |
|---|---|---|
| MLP | ~97% | **74.5%** |
| Random Forest | ~97% | 62.8% |

## Decision
Global accuracy cannot separate the two models: with 97% normal cases, a model that never predicts a failure would already score 97%. The MLP was kept for its higher failure recall, the metric that limits missed failures.

The Random Forest was still useful: its feature importance shows that torque, rotational speed and tool wear carry about 83% of the predictive power.

## Files
- `002_CSVPrediction.py`: first iteration, standalone MLP for failure prediction
- `006_TabularMLPvsRF.py`: full comparison with a Random Forest baseline, interactive menu (train, evaluate, compare)