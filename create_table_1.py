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
sizes_ordered = ["small", "wide", "deep", "default", "large"]
zinc_columns = sizes_ordered
molhiv_columns = ["small"]
release_columns = ["small"]

# All columns in order
columns = zinc_columns + molhiv_columns + release_columns

# Initialize results dictionary
results = {model: {column: "" for column in columns} for model in models}

# Populate the results dictionary
for record in data_list:
    model = record["model"]
    dataset = record["dataset"]
    MAE = record["MAE"]
    size = record["size"]
    if dataset == "zinc":
        column = size
    elif dataset == "molhiv":
        column = "small"  # As per your instruction
    elif dataset == "release":
        column = "small"  # As per your instruction
    else:
        continue
    if column in columns:
        results[model][column] = MAE

# Generate LaTeX table code
# Start constructing the LaTeX table
num_zinc_cols = len(zinc_columns)
num_molhiv_cols = len(molhiv_columns)
num_release_cols = len(release_columns)

# Alignment
alignment = "l" + "c" * (num_zinc_cols + num_molhiv_cols + num_release_cols)

print("\\begin{tabular}{%s}" % alignment)
print("\\hline")

# First header row
header = ["Model"]
header += ["\\multicolumn{%d}{c}{zinc}" % num_zinc_cols]
header += ["\\multicolumn{%d}{c}{molhiv}" % num_molhiv_cols]
header += ["\\multicolumn{%d}{c}{release}" % num_release_cols]
print(" & ".join(header) + " \\\\")

# Second header row
sub_header = [""]
sub_header += zinc_columns
sub_header += molhiv_columns
sub_header += release_columns
print(" & " + " & ".join(sub_header) + " \\\\")
print("\\hline")

# Fill in the data rows
for model in sorted(models):
    row = [model]
    for col in columns:
        value = results[model][col]
        row.append(value)
    print(" & ".join(row) + " \\\\")

print("\\hline")
print("\\end{tabular}")
