import os
import json
import numpy as np

if __name__ == "__main__":

    with open("outputs/eval_generate_results.json", "r") as fp:
        results = json.load(fp)

    # compute the average for each method across all features
    for method_key, method_results in results.items():
        avg = []
        for metric_key, metric_values in method_results.items():
            # compute average for each metric
            metric_avg = sum(metric_values) / len(metric_values)
            results[method_key][metric_key] = metric_avg
            avg.append(metric_avg)
        results[method_key]["avg"] = np.mean(avg)

    table = "Method & RMS & CF & SW & SI & BS & AVG \\\\ \midrule \n"

    metrics = [
        "mix-rms",
        "mix-crest_factor",
        "mix-stereo_width",
        "mix-stereo_imbalance",
        "mix-barkspectrum",
        "avg",
    ]

    methods = {
        "equal_loudness": "Equal Loudness",
        "mst": "MST",
        "diffmst-stft-8": "DiffMST-STFT-8",
        "diffmst-stft-16": "DiffMST-STFT-16",
        "diffmst-stft+AF-8": "DiffMST-STFT+AF-8",
        "diffmst-stft+AF-16": "DiffMST-STFT+AF-16",
        "diffmst-16": "DiffMST-16",
    }

    for method_key, method_name in methods.items():
        row = method_name
        for metric in metrics:
            avg_metric_value = results[method_key][metric]
            row += " & " + f"{avg_metric_value:.2e}"
        row += " \\\\ \n"
        table += row

    print(table)
