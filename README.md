# Chemistry-Aware Topological Neural Networks for Molecular Property Prediction

This project explores the application of Topological Neural Networks (TNNs) to molecular property prediction by leveraging chemical substructures. TNNs are a generalization of Graph Neural Networks that can model higher-order interactions between nodes, making them particularly well-suited for molecular analysis where functional groups and substructures play crucial roles in determining chemical properties.
Our implementation compares six different TNN architectures across three molecular datasets (ZINC-12k, ogbg-molhiv, and ReLeaSE), with a focus on predicting properties like solubility and binding affinity. We utilize the BRICS algorithm for chemical-aware decomposition of molecules into meaningful substructures, which are then lifted into higher-order topological features for processing by our models.

## Requirements

Main dependencies: PyTorch, TopoModelX, RDKit, Weights & Biases

## Models

We implement six TNN architectures:

### Cell Complex Models
- Cell Attention Network (CAN)
- Cellular Complex Network (CCXN)
- Cell Weisfeiler-Lehman Network (CWN)

### Combinatorial Complex Models
- Hierarchical Message Scheme (HMS)

### Hypergraph Models
- Heterogeneous Network (HNHN)
- Universal Simplicial Attention Graph Embedding (UNISAGE)

## Configuration

Each model can be run with five different configurations:
```json
{
    "name": "CAN_large",  # Options: small, default, large, wide, deep
    "model": "CANModel",
    "hidden_dimensions": 40,
    "n_layers": 16,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "test_interval": 5,
    "test_size": 0.2,
    "dataset": "zinc",    # Options: zinc, molhiv, release
    "gradient_accumulation_steps": 400,
    "epoch_size": 5000
}
```

## Running the Code

Using SLURM:
```bash
sbatch launch.slurm
```

Manual execution:
```bash
python src/train.py --config configs/can_large.json 
```

## Experiment Tracking

Results are logged to Weights & Biases. Configure with:
```bash
export WANDB_API_KEY=your_api_key
wandb login
```


## Contact

- Noga Bregman - nogabregman@mail.tau.ac.il
- Jonathan Horovitz - jh1@mail.tau.ac.il
