import re
import csv
import sys

# Input data as a string
data = sys.stdin.read()


# Regex to extract relevant fields with multiline support
pattern = re.compile(
    r"output/(?P<run_number>\d+)\.out.*?"
    r'"name":\s*"(?P<name>[^"]+)",\s*'
    r'"model":\s*"(?P<model>[^"]+)",.*?'
    r"Parameters:\s*(?P<parameters>\d+).*?"
    r"Train Loss:\s*(?P<train_loss>[\d.eE+-]+),\s*"
    r"Test Loss:\s*(?P<test_loss>[\d.eE+-]+),.*?"
    r"MAE:\s*(?P<mae>[\d.eE+-]+)",
    re.DOTALL | re.MULTILINE,
)

# Match the regex and extract data
matches = pattern.finditer(data)

# Prepare rows for the CSV
rows = []
for match in matches:
    rows.append(
        [match.group("run_number"), match.group("name"), match.group("model"), match.group("parameters"), match.group("train_loss"), match.group("test_loss"), match.group("mae")]
    )

# Write to a CSV file
output_csv = "output_data.csv"
with open(output_csv, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header
    csv_writer.writerow(["run_number", "name", "model", "parameters", "train_loss", "test_loss", "MAE"])
    # Write data rows
    csv_writer.writerows(rows)

print(f"Data has been written to {output_csv}")
