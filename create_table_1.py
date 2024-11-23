from collections import defaultdict
import csv


# Function to extract the size from the name field
def extract_size(name):
    parts = name.split("_")
    if len(parts) >= 2:
        size = parts[1]
    else:
        size = ""
    return size


# Initialize data structures
models = set()
sizes = set()
data_list = []

# Read the CSV file
with open("output_data.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header
    for row in reader:
        run_number, name, model, dataset, max_epoch, parameters, MAE = row
        models.add(model)
        size = extract_size(name)
        sizes.add(size)
        data_list.append({"model": model, "name": name, "dataset": dataset, "MAE": MAE, "size": size})

# Define the order of sizes for consistent column ordering
zinc_columns = ["small", "default", "large", "wide", "deep"]
molhiv_columns = ["small"]
release_columns = ["small"]

# All columns in order
columns = zinc_columns + molhiv_columns + release_columns

results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
models = set()

# Populate the results dictionary
for record in data_list:
    model = record["model"]
    dataset = record["dataset"]
    MAE = record["MAE"]
    size = record["size"]
    models.add(model)
    if dataset == "zinc":
        column = size
    elif dataset in ["molhiv", "release"]:
        column = "small"  # As per your instruction
    else:
        continue
    results[model][dataset][column] = MAE

# Build alignment, headers, and table columns dynamically
datasets = [("zinc", zinc_columns), ("molhiv", molhiv_columns), ("release", release_columns)]

alignment = "l"  # For the 'Model' column
header = ["Model"]
sub_header = [""]
table_columns = []

for dataset_name, dataset_columns in datasets:
    num_cols = len(dataset_columns)
    alignment += "c" * num_cols
    header.append("\\multicolumn{%d}{c}{%s}" % (num_cols, dataset_name))
    sub_header.extend(dataset_columns)
    table_columns.extend([(dataset_name, col) for col in dataset_columns])

# Generate LaTeX table code
print("\\begin{tabular}{%s}" % alignment)
print("\\hline")
print(" & ".join(header) + " \\\\")
print(" & " + " & ".join(sub_header) + " \\\\")
print("\\hline")

# Fill in the data rows
for model in sorted(models):
    row = [model]
    for dataset_name, col in table_columns:
        value = results[model][dataset_name].get(col, "")
        row.append(str(value))
    print(" & ".join(row) + " \\\\")

print("\\hline")
print("\\end{tabular}")
