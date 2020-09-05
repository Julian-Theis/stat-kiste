# Stat Kiste
 
The *Stat Kiste* is a collection of statistical methods that I used throughout the last few years.
Each method can be called from the command line. 

Currently, the *Stat Kiste* contains the following implementations:
- <b>AUC with Confidence Intervals</b>: Fast DeLong method, implementation by the Yandex Data School [https://github.com/yandexdataschool/roc_comparison](https://github.com/yandexdataschool/roc_comparison)


## How-To
### AUC with Confidence Intervals
Calculating the AUC with Confidence Intervals based on a given alpha and a csv-file that contains y_pred and corresponding y_true values.
```python
python run.py --function ci_auc -f <path_to_csv_file> -ypc <y_pred_column> -ytc <y_true_column> -a <alpha>
```

Example source code is provided in [examples/example_ci_auc.py](examples/example_ci_auc.py).
An example csv file is provided in [examples/ci_auc_test.csv](examples/ci_auc_test.csv).

