def log_predictions(fold, dataset_name, y_test, yhat_point, mu, std, quantile_preds, file_path="logs/openml/predictions_PBGM.csv"):
    """Logs the predictions to a CSV file."""
    header = ["fold", "dataset", "true_value", "predicted_value", "mu", "std", "quantile_0.1", "quantile_0.5", "quantile_0.9"]

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    # Open the file in append mode ('a+')
    with open(file_path, mode='a+', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file does not exist or is empty
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(header)

        # Write each prediction for the current fold
        for i in range(len(y_test)):
            row_to_write = [
                fold,
                dataset_name,
                y_test.iloc[i],
                yhat_point[i],
                mu[i],
                std[i],
                quantile_preds['0.1'][i],
                quantile_preds['0.5'][i],
                quantile_preds['0.9'][i]
            ]
            writer.writerow(row_to_write)
