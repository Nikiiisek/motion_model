import csv
from pathlib import Path


def save_experiment_to_csv(csv_path, experiment_data):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=experiment_data.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(experiment_data)