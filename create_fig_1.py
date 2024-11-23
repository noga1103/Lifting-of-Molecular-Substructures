import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set Seaborn theme
sns.set_theme(style="whitegrid")


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
        data_list.append({"model": model, "dataset": dataset, "size": size, "MAE": float(MAE)})

# Define datasets and sizes order
datasets = ["zinc", "molhiv", "release"]
sizes_ordered = ["small", "wide", "deep", "default", "large"]

# Prepare data for plotting
plot_data = []
for record in data_list:
    dataset_size = f"{record['dataset']}_{record['size']}"
    plot_data.append((record["model"], dataset_size, record["MAE"], record["dataset"], record["size"]))

# Convert data into a grouped structure
models = sorted(models)
dataset_size_combos = sorted({item[1] for item in plot_data})

# Map dataset and size combinations to distinct colors
zinc_colors = {
    "small": "blue",
    "wide": "cyan",
    "deep": "purple",
    "default": "green",
    "large": "pink",
}
dataset_colors = {
    "molhiv": "orange",
    "release": "red",
}
combo_colors = {
    **{f"zinc_{size}": zinc_colors[size] for size in zinc_colors},
    **{f"{dataset}_small": dataset_colors[dataset] for dataset in ["molhiv", "release"]},
}

# Plotting
plt.figure(figsize=(12, 6))

group_spacing = 0.5  # Space between groups
bar_width = 0.15
x_positions = np.arange(len(models)) * (len(dataset_size_combos) * bar_width + group_spacing)


for j, dataset_size in enumerate(dataset_size_combos):
    values = [
        next(
            (item[2] for item in plot_data if item[0] == model and item[1] == dataset_size),
            0,
        )
        for model in models
    ]
    plt.bar(
        x_positions + j * bar_width,
        values,
        width=bar_width,
        color=combo_colors.get(dataset_size, "gray"),
        label=dataset_size if j < len(dataset_size_combos) else None,
    )

# Formatting
plt.ylim(0, 2)
plt.xticks(x_positions + (len(dataset_size_combos) - 1) * bar_width / 2, models, rotation=45)
plt.ylabel("MAE")
plt.xlabel("Models")
plt.title("MAE Comparison Across Models and Dataset/Size Combinations")
plt.legend(title="Dataset_Size", loc="upper left", bbox_to_anchor=(1.05, 1))
plt.tight_layout()

# Save the plot
plt.savefig("fig_1.png", dpi=300, bbox_inches="tight")
