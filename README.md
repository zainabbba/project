# Aircraft Engine Remaining Useful Life Prediction

## Project Description

This project predicts the remaining useful life (RUL) of simulated aircraft engines using the current operating cycle, operating settings, and sensor readings. RUL estimates how many operating cycles remain before failure and can support predictive maintenance planning.

The project includes exploratory data analysis, preprocessing, four regression algorithms, grouped cross-validation, hyperparameter tuning, a comparison of full and capped RUL, and a Streamlit application.

## Dataset

- **Source:** NASA Prognostics Center of Excellence (PCoE), *Turbofan Engine Degradation Simulation Data Set*, item 6 in the [NASA data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
- **Download:** [NASA dataset ZIP](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip).
- **Subset:** FD001 from the C-MAPSS simulator.
- **Training set:** 20,631 observations from 100 engines.
- **Test set:** 100 engines with trajectories ending before failure.
- **Observation:** Each row represents one engine at one operating cycle.
- **Scope:** One operating condition and one fault mode involving high-pressure compressor degradation.
- **Target:** Remaining useful life, measured in operating cycles.

These are simulated engine degradation trajectories, rather than measurements collected directly from aircraft in service.

### Features

| Feature | Description |
|---|---|
| `engine_id` | Engine identifier used for tracking and grouped validation; excluded from model inputs. |
| `cycle` | Number of operating cycles completed at the time of observation. |
| `setting_1`–`setting_3` | Three operational settings describing engine operating conditions. |
| `sensor_1`–`sensor_21` | Measurements representing temperatures, pressures, rotational speeds, flow-related quantities, and other engine parameters. |

Examples of documented sensor meanings:

| Feature | Measurement | Unit |
|---|---|---|
| `sensor_2` | Total temperature at the low-pressure compressor outlet | °R |
| `sensor_3` | Total temperature at the high-pressure compressor outlet | °R |
| `sensor_4` | Total temperature at the low-pressure turbine outlet | °R |
| `sensor_7` | Total pressure at the high-pressure compressor outlet | psia |

Sensor definitions are documented in [Table 1 of the original NASA paper](https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf). Temperature values use degrees Rankine, and pressure values use pounds per square inch absolute.

### Target Definition

For each run-to-failure training trajectory:

```python
RUL = final_recorded_cycle - current_cycle
```

For the capped formulation:

```python
RUL_capped = RUL.clip(upper=125)
```

Full RUL values of 180 and 220 both become 125, while values of 20 and 40 remain unchanged. A capped target value of 125 represents **125 or more cycles remaining**.

Capping removes distinctions beyond the chosen horizon while preserving differences within the final 125 cycles. The threshold is a modeling choice, not a physical failure boundary or a universal maintenance standard.

For the test engines, NASA supplies the true RUL at the last observed cycle in `RUL_FD001.txt`. Evaluation of the capped target uses these reference values capped at the same threshold.

## Repository Structure

```text
project/
├── Data/
│   ├── Original/       # Original dataset files and documentation
│   └── Cleaned/        # Data produced by preprocessing
├── Notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   └── 02_Modeling_and_Evaluation.ipynb
├── Models/             # Saved model pipelines
├── Helper/             # Supporting Python functions
├── main.py             # Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

## Setup Instructions

The project environment uses Python 3.14.0. Ensure the dependencies in `requirements.txt` support your Python version.

Open a terminal in the project root, where `requirements.txt` and `main.py` are located.

### 1. Create and Activate a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS and Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download and extract the NASA archive. Place these files directly in `Data/Original/`:

- `train_FD001.txt` — Training trajectories that run to failure.
- `test_FD001.txt` — Test trajectories that end before failure.
- `RUL_FD001.txt` — Reference RUL values for the test engines.

## Run the Notebooks

From the project root, with the virtual environment activated:

```bash
python -m jupyter notebook
```

Run all cells in `Notebooks/01_EDA_and_Preprocessing.ipynb` first to prepare the data. Then run `Notebooks/02_Modeling_and_Evaluation.ipynb` to train, evaluate, and save model pipelines as `.joblib` files in `Models/`.

## Run the Streamlit App

After generating the required model files, run:

```bash
python -m streamlit run main.py
```

Open the local URL displayed in the terminal.

The interpretation of a prediction depends on the loaded model. A full-RUL model estimates unrestricted remaining cycles, while a capped-RUL model estimates the capped target. The app label should identify which formulation is being used.

## Work Summary

1. Inspected data types, missing values, duplicate rows, engine lifespans, and the RUL distribution. Examined sensor trends, statistical outliers, and feature correlations.
2. Calculated training RUL. Removed seven constant columns and excluded the engine identifier from model inputs. Retained statistical outliers and used 18 input features.
3. Compared Linear Regression, Ridge (`alpha=1.0`), Random Forest, and Gradient Boosting. Applied `StandardScaler` inside the two linear model pipelines; tree-based models used unscaled inputs.
4. Used five-fold `GroupKFold`, keeping all observations from each engine together in a fold. Tuned Random Forest and Gradient Boosting with grouped cross-validation and ranked configurations by mean validation RMSE.
5. Evaluated models on the official FD001 test set using the last observed row of each test engine and its supplied RUL.
6. Added a comparison of full RUL and RUL capped at 125 cycles.
7. Saved fitted pipelines and built a Streamlit interface for estimating RUL from engine inputs.

### Constant Columns Removed

```text
sensor_1, setting_3, sensor_5, sensor_10,
sensor_18, sensor_19, sensor_16
```

Statistical outliers were retained because unusual sensor readings may reflect engine degradation rather than errors.

**Target capping is separate from sensor outlier handling:** it changes the target being predicted rather than clipping sensor readings.

## Modeling and Evaluation

### Grouped Validation

Five-fold `GroupKFold` uses `engine_id` as the grouping variable. Each engine appears in either the training or validation portion of a fold, preventing observations from the same engine from appearing in both.

### Hyperparameter Tuning

Random Forest and Gradient Boosting were tuned with `GridSearchCV`. Configurations were ranked by mean validation RMSE.

Scikit-learn uses negative error scores during the search. Reported RMSE values are converted back to positive numbers.

Model configurations should be selected using training cross-validation, with the official test set reserved for final reporting.

### Test Evaluation

Evaluation uses one observation per test engine: its last recorded operating cycle. Predictions are compared with the corresponding values in `RUL_FD001.txt`.

For capped evaluation, the reference target is capped at 125 cycles. Full and capped targets must be reported separately because they define different prediction tasks.

## Results

### Initial Full-RUL Experiment

| Model | MAE (cycles) | RMSE (cycles) | R² |
|---|---:|---:|---:|
| Gradient Boosting | 18.831 | 25.226 | 0.632 |
| Random Forest | 19.139 | 25.606 | 0.620 |
| Ridge | 25.975 | 31.250 | 0.434 |
| Linear Regression | 25.975 | 31.250 | 0.434 |

These results belong to the initial experiment. The following table reports the updated comparison supplied for the two tree-based models.

### Updated Full-RUL and Capped-RUL Comparison

| Model | R² (Full RUL) | R² (Capped at 125) | MAE (Full RUL) | MAE (Capped at 125) | RMSE (Full RUL) | RMSE (Capped at 125) |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | **0.639** | **0.831** | **17.993** | **11.943** | **24.970** | **16.480** |
| Random Forest | 0.586 | 0.826 | 20.257 | 12.398 | 26.733 | 16.735 |

MAE and RMSE are measured in operating cycles.

**Best observed test performance: Gradient Boosting.** It achieved the lowest MAE and RMSE and the highest R² among the two models in both updated formulations.

For capped RUL, its predictions differed from the capped reference values by **11.943 cycles on average in absolute terms**.

### Interpretation

The capped experiment produced lower errors and higher R² values. However, capping changes the target distribution and removes distinctions above 125 cycles. These metrics therefore do not establish better prediction of unrestricted remaining life.

- **Full RUL:** Estimates remaining cycles across the complete lifespan.
- **Capped RUL:** Estimates remaining life within the chosen 125-cycle horizon.

A prediction at the cap should be interpreted as approximately **125 or more cycles remaining**, rather than exactly 125 cycles.

### Evaluation Metrics

- **MAE:** Average absolute prediction error.
- **RMSE:** Prediction error with greater weight assigned to large errors.
- **R²:** Compares the model's squared errors with those from predicting the mean true target value of the evaluation set.

## Conclusions and Limitations

- Sensor readings and operating-cycle information support RUL estimation on simulated FD001 engine trajectories.
- Gradient Boosting achieved the strongest test metrics among the two models in the updated comparison.
- Capping at 125 cycles produced stronger metrics for the capped prediction task.
- FD001 covers one operating condition and one fault mode, limiting generalization to more varied conditions.
- Simulated-data results do not establish readiness for real-world aircraft maintenance decisions.
- Predictions do not currently include uncertainty intervals.
- Aggregate metrics do not reveal whether errors are concentrated near failure.
- Repeated use of test results to guide modeling choices can make reported performance optimistic.

## Future Improvements

- Add rolling sensor statistics and temporal trend features.
- Analyze prediction errors specifically near failure.
- Investigate overestimation of remaining life and report uncertainty intervals.
- Compare capping thresholds using grouped training validation.
- Perform data-driven feature selection inside cross-validation.
- Evaluate additional C-MAPSS subsets with more operating conditions and fault modes.
- Keep saved model target definitions consistent with Streamlit prediction labels.

## References

- Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). [Damage Propagation Modeling for Aircraft Engine Prognostics](https://ntrs.nasa.gov/citations/20090029214).
- [Scikit-learn: GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [Scikit-learn: GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Scikit-learn: Regression Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [Streamlit documentation](https://docs.streamlit.io/)
