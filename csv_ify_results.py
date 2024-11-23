import re
import csv
import sys

# Input data as a string
data = sys.stdin.read()


pattern = re.compile(
    r"""output_keep/(?P<run_number>\d+)\.out\s*
\s*"name":\s*"(?P<name>[^"]+)",\s*
\s*"model":\s*"(?P<model>[^"]+)",\s*
\s*"dataset":\s*"(?P<dataset>[^"]+)",\s*
(?:\s*"learning_rate":\s*[^,]+,\s*)?
Parameters:\s*(?P<parameters>\d+)\s*
Epoch:(?P<epoch>\d+),.*?MAE:\s*(?P<MAE>[^,]+),"""
)


# Match the regex and extract data
matches = pattern.finditer(data)

# Prepare rows for the CSV
rows = []
for match in matches:
    rows.append(
        [
            match.group("run_number"),
            match.group("name"),
            match.group("model"),
            match.group("dataset"),
            match.group("epoch"),
            match.group("parameters"),
            match.group("MAE"),
        ]
    )

# Write to a CSV file
output_csv = "output_data.csv"
with open(output_csv, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header
    csv_writer.writerow(
        [
            "run_number",
            "name",
            "model",
            "dataset",
            "max epoch",
            "parameters",
            "train_loss",
            "test_loss",
            "MAE",
        ]
    )
    # Write data rows
    csv_writer.writerows(rows)

print(f"Data has been written to {output_csv}")
