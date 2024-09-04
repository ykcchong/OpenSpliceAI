# extracts all the benchmarking results, assuming we're in the benchmark_predict dir

import json
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def generate_paths(subsets, dataset, use_anno):
    path_list = []
    for subset in subsets:
        base_dir = f'results_{subset}'
        for model in ['keras', 'pytorch']:
            for flanking_size in [80, 400, 2000, 10000]:
                parent_dir = f'{model}_{dataset}_sub{subset}_{flanking_size}nt_anno{use_anno}'
                path_list.append([os.path.join(base_dir, parent_dir, f'trial_{trial}', 'profile.json') for trial in [1, 2, 3, 4, 5]])
    return path_list

def extract_jsons(paths):
    data = []
    for group in paths:
        for path in group:
            # skip if the scalene was not able to create the file
            if not os.path.exists(path):
                print(f"[INFO]: {path} does not exist")
                continue

            # load the json file
            with open(path, 'r') as f:
                json_data = json.load(f)

                # handle different version
                if 'keras' in path:
                    for line in json_data['files']['/ccb/cybertron/smao10/openspliceai/experiments/benchmark_predict/spliceai_default_test.py']['functions']:
                        if line['line'] == 'predict_and_write':
                            predict_data = line
                elif 'pytorch' in path:
                    for line in json_data["files"]["/ccb/cybertron/smao10/openspliceai/experiments/benchmark_predict/predict_test.py"]["functions"]:
                        if line['line'] == 'predict_and_write':
                            predict_data = line
                else:
                    print("[ERROR] Malformed path!")
                    sys.exit(1)

                # handle the test parameters
                parts = path.split('/')
                trial_num = int(parts[-2].split('_')[1])
                params = parts[-3].split('_')
                model_type = params[0]
                subset_size = int(params[2][3:])
                flanking_size = int(params[3][:-2])
            
                row = {
                    'subset_size': subset_size,
                    'model_type': model_type,
                    'flanking_size': flanking_size,
                    'trial_num': trial_num,
                    "elapsed_time_sec": json_data.get("elapsed_time_sec", 0),
                    "growth_rate": json_data.get("growth_rate", 0),
                    "max_footprint_mb": json_data.get("max_footprint_mb", 0),
                    "n_avg_mb": predict_data.get("n_avg_mb", 0),
                    "n_copy_mb_s": predict_data.get("n_copy_mb_s", 0),
                    "n_core_utilization": predict_data.get("n_core_utilization", 0),
                    "n_cpu_percent_c": predict_data.get("n_cpu_percent_c", 0),
                    "n_cpu_percent_python": predict_data.get("n_cpu_percent_python", 0),
                    "n_gpu_avg_memory_mb": predict_data.get("n_gpu_avg_memory_mb", 0),
                    "n_gpu_peak_memory_mb": predict_data.get("n_gpu_peak_memory_mb", 0),
                    "n_gpu_percent": predict_data.get("n_gpu_percent", 0),
                    "n_growth_mb": predict_data.get("n_growth_mb", 0),
                    "n_malloc_mb": predict_data.get("n_malloc_mb", 0),
                    "n_mallocs": predict_data.get("n_mallocs", 0),
                    "n_peak_mb": predict_data.get("n_peak_mb", 0),
                    "n_python_fraction": predict_data.get("n_python_fraction", 0),
                    "n_sys_percent": predict_data.get("n_sys_percent", 0),
                    "n_usage_fraction": predict_data.get("n_usage_fraction", 0),
                    "samples": json_data.get("samples", 0)
                }
                data.append(row)

    return pd.DataFrame(data)


def fit_line_of_best_fit(samples):
    # Extract all points from the samples
    all_points = np.concatenate(samples)
    X = all_points[:, 0].reshape(-1, 1)
    y = all_points[:, 1]
    
    # Fit a linear regression model
    model = LinearRegression().fit(X, y)
    
    # Predict y values for the given X values
    y_pred = model.predict(X)
    
    # Return the line of best fit as a list of points
    return list(zip(X.flatten(), y_pred))

def calculate_averages(dataframe):
    grouped = dataframe.groupby(['subset_size', 'model_type', 'flanking_size'])
    
    averages_list = []
    for name, group in grouped:
        average_row = group.mean(numeric_only=True).to_dict()
        average_row['subset_size'], average_row['model_type'], average_row['flanking_size'] = name
        
        # Collect all sample points
        samples = group['samples'].tolist()
        
        # Fit line of best fit
        line_of_best_fit = fit_line_of_best_fit(samples)
        
        # Store the line of best fit in the average row
        average_row['samples'] = line_of_best_fit
        
        averages_list.append(average_row)
    
    averages = pd.DataFrame(averages_list)
    return averages


def write_data_to_file(dataframe, averages):
    with open('aggregated_results.txt', 'w') as f:
        dataframe.to_string(f, index=False)

    with open('averaged_results.txt', 'w') as f:
        averages.to_string(f, index=False)

    with open('aggregated_results.csv', 'w') as f:
        dataframe.to_csv(f, index=False)

    with open('averaged_results.csv', 'w') as f:
        averages.to_csv(f, index=False)
            
def main():
    subsets = [50, 100, 200, 300, 400, 500, 1000]
    use_anno = 1
    dataset = 'MANE'

    path_list = generate_paths(subsets, dataset, use_anno)
    df = extract_jsons(path_list)
    print(df.head())
    averages = calculate_averages(df)
    write_data_to_file(df, averages)
    print('Data and averages written')

if __name__ == '__main__':
    main()
