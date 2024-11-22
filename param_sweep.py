import os
import json

INPUT_DIR = "./configs/small"
OUTPUT_DIR = "./configs/param_sweep_4"


def update_config(config):
    """Update the learning_rate key in the config for each rate and return modified configs."""
    updated_configs = []

    molhiv_config = config.copy()
    molhiv_config["name"] = molhiv_config["name"] + "_molhiv"
    molhiv_config["num_epochs"] = 50
    molhiv_config["dataset"] = "molhiv"
    updated_configs.append(molhiv_config)

    release_config = config.copy()
    release_config["name"] = release_config["name"] + "_release"
    release_config["num_epochs"] = 50
    release_config["dataset"] = "release"
    updated_configs.append(release_config)

    return updated_configs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    for json_file in json_files:
        input_path = os.path.join(INPUT_DIR, json_file)
        with open(input_path, "r") as f:
            config = json.load(f)

        updated_configs = update_config(config)

        # Save
        for updated_config in updated_configs:
            output_path = os.path.join(OUTPUT_DIR, updated_config["name"] + ".json")

            with open(output_path, "w") as f:
                json.dump(updated_config, f, indent=4)


if __name__ == "__main__":
    main()
