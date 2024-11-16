import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from train.can import CANModel
from train.ccxn import CCXNModel
from train.cwn import CWNModel

# from train.hmc import HMCModel
from train.train_utils import DEVICE, WEIGHT_DTYPE, load_molhiv_data, load_zinc_data_small, load_zinc_data
import json
import sys
import os
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", f"local_{random.randint(0, 100000)}")


def load_config(config_path):
    with open(config_path, "r") as f:
        config_str = f.read()
        print(f"Running job {SLURM_JOB_ID} with config: {config_path}")
        print(config_str)
        config = json.loads(config_str)
    return config


def initialize_model(config):
    HIDDEN_DIMENSIONS = config["hidden_dimensions"]
    if config["model"] == "CCXNModel":
        model = CCXNModel(HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS, n_layers=config["n_layers"])
    elif config["model"] == "CANModel":
        model = CANModel(HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS, n_layers=config["n_layers"])
    # elif config["model"] == "HMCModel":
    #     model = HMCModel(HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS, n_layers=config["n_layers"])
    elif config["model"] == "CWNModel":
        model = CWNModel(HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS, HIDDEN_DIMENSIONS, n_layers=config["n_layers"])
    else:
        raise ValueError("Unknown model: {}".format(config["model"]))
    model = model.to(DEVICE)
    return model


def prepare_data(config, model):
    if config["dataset"] == "molhiv":
        data = load_molhiv_data()
    elif config["dataset"] == "zinc":
        data = load_zinc_data()
    elif config["dataset"] == "zinc_small":
        data = load_zinc_data_small()
    else:
        raise ValueError("Unknown dataset: {}".format(config["dataset"]))

    [model.add_graph_matrices(graph) for graph in data]
    return data


def plot_metrics(metrics_list, output_dir):
    steps = [m["step"] for m in metrics_list]

    # Plot train_loss and test_loss together
    train_losses = [m["train_loss"] for m in metrics_list]
    test_losses = [m["test_loss"] for m in metrics_list]
    plt.figure()
    plt.plot(steps, train_losses, label="Train Loss", marker="o")
    plt.plot(steps, test_losses, label="Test Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plot_file = os.path.join(output_dir, "loss.png")
    plt.savefig(plot_file)
    plt.close()

    # Plot other metrics
    metrics_names = ["r2", "mae", "rmse"]
    for metric_name in metrics_names:
        values = [m[metric_name] for m in metrics_list]
        plt.figure()
        plt.plot(steps, values, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.upper())
        plt.title(f"{metric_name.upper()} over Epochs")
        plt.grid(True)
        plot_file = os.path.join(output_dir, f"{metric_name}.png")
        plt.savefig(plot_file)
        plt.close()


def train_model(model, train_data, test_data, config, output_dir):
    start = datetime.datetime.now()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    test_interval = config["test_interval"]
    num_epochs = config["num_epochs"]
    metrics_list = []

    for epoch_i in range(1, num_epochs + 1):
        epoch_loss = []
        model.train()
        optimizer.zero_grad()
        for _ in tqdm(range(config["epoch_size"]), desc=f"Epoch {epoch_i}/{num_epochs} Training", unit="graph"):
            losses = []
            graph = random.choice(train_data)
            y = torch.tensor([graph.regression_value], dtype=WEIGHT_DTYPE).to(DEVICE)
            y_hat = model(graph)
            loss = loss_fn(y_hat, y)
            losses.append(loss)
            if len(losses) == config["gradient_accumulation_steps"]:
                loss = torch.stack(losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss.append(loss.item())

        if epoch_i % test_interval == 0:
            model.eval()
            y_true_list, y_pred_list = [], []
            with torch.no_grad():
                train_mean_loss = np.mean(epoch_loss)
                test_losses = []
                current_test_data = random.sample(test_data, min(config["epoch_size"], len(test_data)))
                for graph in tqdm(current_test_data, desc=f"Epoch {epoch_i}/{num_epochs} Testing", unit="graph"):
                    y = torch.tensor([graph.regression_value], dtype=WEIGHT_DTYPE).to(DEVICE)
                    y_hat = model(graph)
                    test_loss = loss_fn(y_hat, y)
                    y_true_list.append(y.item())
                    y_pred_list.append(y_hat.item())
                    test_losses.append(test_loss.item())

                test_mean_loss = np.mean(test_losses)
                r2 = r2_score(y_true_list, y_pred_list)
                mae = mean_absolute_error(y_true_list, y_pred_list)
                rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))
                print(f"Epoch:{epoch_i}, Train Loss: {train_mean_loss:.4f}, Test Loss: {test_mean_loss:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                metrics = {"step": epoch_i, "train_loss": train_mean_loss, "test_loss": test_mean_loss, "r2": r2, "mae": mae, "rmse": rmse}
                metrics_list.append(metrics)

    # Save metrics as JSON
    metrics_file = os.path.join(output_dir, "metrics.json")
    end = datetime.datetime.now()
    metrics_dict = {"runtime_metrics": metrics_list, "parameters": count_parameters(model), "training_start": start, "training_end": end, "elapsed": end - start}
    with open(metrics_file, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # Plot and save graphs for each metric
    plot_metrics(metrics_list, output_dir)

    # Save the model
    model_file = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), model_file)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    config_path = sys.argv[1]
    config = load_config(config_path)
    torch.manual_seed(0)
    model = initialize_model(config)
    print(f"Parameters: {count_parameters(model)}")
    full_data = prepare_data(config, model)
    print(f"Data loaded: {len(full_data)}")
    output_dir = f"results/{config['name']}_{SLURM_JOB_ID}/"
    os.makedirs(output_dir, exist_ok=True)
    train_data, test_data = train_test_split(full_data, test_size=config["test_size"], shuffle=True)
    print("Starting training...")
    train_model(model, train_data, test_data, config, output_dir=output_dir)


if __name__ == "__main__":
    main()
