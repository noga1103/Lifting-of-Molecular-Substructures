# Chemistry-Aware Topological Neural Networks for Molecular Property Prediction

Implementation of Topological Neural Networks (TNNs) for molecular property prediction, focusing on chemistry-aware lifting of molecular substructures.

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
