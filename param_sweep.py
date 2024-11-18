import os
import json

INPUT_DIR = "./configs/default"
OUTPUT_DIR = "./configs/param_sweep_2"
MULTIPLES = [(1, 1, "default"), (0.5, 2, "wide"), (2, 0.5, "deep"), (2, 2, "large"), (0.5, 0.5, "small")]  # n-layers, hidden_dim, name


def update_learning_rate(config):
    """Update the learning_rate key in the config for each rate and return modified configs."""
    updated_configs = []
    for layers_multiple, hidden_dim_multiple, name in MULTIPLES:
        updated_config = config.copy()
        updated_config["n_layers"] = int(updated_config["n_layers"] * layers_multiple)
        updated_config["hidden_dimensions"] = int(updated_config["hidden_dimensions"] * hidden_dim_multiple)
        updated_config["name"] = f"{config['name']}_{name}"
        updated_configs.append(updated_config)
    return updated_configs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    for json_file in json_files:
        input_path = os.path.join(INPUT_DIR, json_file)
        with open(input_path, "r") as f:
            config = json.load(f)

        updated_configs = update_learning_rate(config)

        # Save
        for updated_config in updated_configs:
            output_path = os.path.join(OUTPUT_DIR, updated_config["name"] + ".json")

            with open(output_path, "w") as f:
                json.dump(updated_config, f, indent=4)


if __name__ == "__main__":
    main()
