import os
import json

INPUT_DIR = "./configs/param_sweep_2"
OUTPUT_DIR = "./configs/param_sweep_2"


def update_learning_rate(config):
    """Update the learning_rate key in the config for each rate and return modified configs."""
    updated_configs = []
    updated_config = config.copy()
    updated_config["learning_rate"] = 0.001
    updated_config["gradient_accumulation_steps"] = 400
    updated_config["num_epochs"] = 100
    updated_config["epoch_size"] = 5000
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
