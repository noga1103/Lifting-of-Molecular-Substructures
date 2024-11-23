import json
import os
import matplotlib.pyplot as plt

# Define the directory structure and file paths
results_dirs = {
    "CAN": "CAN_small_100459",
    "CCXN": "CCXN_small_100464",
    "CWN": "CWN_small_100469",
    "HMC": "HMC_small_100474",
    "HNHN": "HNHN_small_107081",
    "UNISAGE": "UNISAGE_small_108518",
}

base_path = "results"
metrics_data = {}

# Load data from metrics.json files
for key, value in results_dirs.items():
    file_path = os.path.join(base_path, value, "metrics.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            metrics_data[key] = json.load(f)
    else:
        print(f"File not found: {file_path}")
        continue

# Prepare data for plotting
loss_data = {}
elapsed_times = {}

for key, data in metrics_data.items():
    runtime_metrics = data["runtime_metrics"]
    steps = [entry["step"] for entry in runtime_metrics]
    test_losses = [entry["test_loss"] for entry in runtime_metrics]
    loss_data[key] = (steps, test_losses)
    elapsed_times[key] = data["elapsed"]

# Plot test_loss by step
plt.figure(figsize=(10, 6))
for key, (steps, test_losses) in loss_data.items():
    plt.plot(steps, test_losses, label=key)
plt.xlabel("Step")
plt.ylabel("Test Loss")
plt.title("Test Loss by Step")
plt.legend()
plt.grid()
plt.savefig("fig_2.png")
plt.close()

# Plot elapsed time as a bar graph
plt.figure(figsize=(10, 6))
plt.bar(elapsed_times.keys(), elapsed_times.values())
plt.xlabel("Key")
plt.ylabel("Elapsed Time (seconds)")
plt.title("Elapsed Time by Model")
plt.grid(axis="y")
plt.savefig("fig_3.png")
plt.close()

print("Graphs have been saved as fig_2.png and fig_3.png.")
