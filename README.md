# AML - Uncertainty Estimation For Deep Ensembles

This is the repository for the "MLMI4: Advanced Machine Learning" coursework of Group 6 on reproducing the results of the "Simple and scalable predictive uncertainty estimation using deep ensembles" paper [1].

## Installation

Create a Python environment with Conda:

```bash
conda create -n deep_ensembles python=3.11
```

Activate the Conda environment:

```bash
conda activate deep_ensembles
```

Install the requirements inside the environment:

```bash
pip install -r requirements.txt
```

If using wandb, it is recommended to create a `.env` file for the variables:

```bash
export WANDB_API_KEY=<your_api_key_here>
export WANDB_ENTITY=<your_entity_name_here> #AML_group6
export WANDB_PROJECT=<your_project_name_here> #AML_Deep_ensembles
```

And then source it:

```bash
source .env
```

## Classification

Train a deep ensemble of classifiers on MNIST:

```bash
python run.py --wandb --run_name mnist_ensemble --dataset mnist --classification_or_regression classification  --number_of_models 5
```

### Experiment on changing the number of models in the ensemble

Run the experiment and measure metric changes when altering the number of models in the ensemble:

```bash
python run.py --wandb --run_name mnist_experiment_with_different_model_numbers --dataset mnist --classification_or_regression classification  --number_of_models 15 --perform_ensemble_experiment
```

### Baseline experiment on changing the number of inference steps in MC dropout

Run the baseline experiment and measure metric changes when varying the number of inference steps per sample in MC dropout:

```bash
python run.py --wandb --run_name mnist_mc_dropout_experiment --dataset mnist --classification_or_regression classification  --number_of_steps 15 --perform_mc_dropout_experiment
```

## Regression

## References

[1] Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in neural information processing systems 30 (2017).
