# Stat Kiste
 
The *Stat Kiste* is a collection of statistical methods that I used throughout the last few years.
Each method can be called from the command line. 

Currently, the *Stat Kiste* contains the following implementations:
- <b>AUC with Confidence Intervals</b>: Fast DeLong method for binary classification, implementation by the Yandex Data School [https://github.com/yandexdataschool/roc_comparison](https://github.com/yandexdataschool/roc_comparison)
- <b>Extended Classification Report</b>: Report for multiclass classification that calculates the accuracy, precision, recall, f1-score, support, predicted values, AUC (sklearn), AUC (DeLong), AUC COV, and AUC Confidence Intervals based on a given alpha value for one-vs-all.
Additionally, the report lists the average/total for precision, recall, f-1 and accuracy as well as the overall AUC (sklearn) and the sum of support and predicted values.

## How-To
### AUC with Confidence Intervals
Calculating the AUC with Confidence Intervals based on a given alpha and a csv-file that contains y_score and corresponding y_true values from a binary classification problem. Prints a confusion matrix.
```python
python run.py --function ci_auc -f <path_to_csv_file> -ysc <y_score_column> -ytc <y_true_column> -a <alpha>
```

Example source code is provided in [examples/example_ci_auc.py](examples/example_ci_auc.py).

An example csv file is provided in [examples/data_ci_auc.csv](examples/data_ci_auc.csv).

### Extended Classification Report
Run the extended Classification Report based on a csv-file that contains y_pred, y_score, and corresponding y_true values from a binary or multiclass classification problem. Prints a confusion matrix.
```python
python run.py --function classification_report -f <path_to_csv_file> -ysc <y_score_column> -ytc <y_true_column> -ypc <y_pred_column> -a <alpha>
```

Example source code is provided in [examples/example_classification_report.py](examples/example_classification_report.py).

An example csv file is provided in [examples/data_classification_report.csv](examples/data_classification_report.csv).

