import torch
from evaluation import (
    get_predictions_and_targets,
    get_predictions_and_targets_mc_dropout,
    compute_predictive_entropy,
)
from models import ClassificationEnsemble


def load_saved_model(path, device):
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()

    return model

def load_saved_ensemble(model_paths, device):
    models = [load_saved_model(path, device) for path in model_paths]
    ensemble = ClassificationEnsemble(models).to(device)
    ensemble.eval()

    return ensemble

def get_mnist_checkpoint_paths():
    ensemble_paths = [
        f"models/classification/mnist/model_mnist_{i}.pt"
        for i in range(1, 16)
    ]
    mc_dropout_path = "models/classification/mnist/model_mnist_mc_dropout.pt"
    return ensemble_paths, mc_dropout_path

def compute_entropy_comparison_results(
    mnist_test_loader,
    notmnist_loader,
    device,
    ensemble_sizes=(1, 5, 10),
):
    ensemble_paths, mc_dropout_path = get_mnist_checkpoint_paths()

    experiment_results = {}

    # Deep Ensembles
    for M in ensemble_sizes:
        print(f"Running Ensemble M={M}")
        ensemble_model = load_saved_ensemble(ensemble_paths[:M], device)

        mnist_probs, _ = get_predictions_and_targets(ensemble_model, mnist_test_loader, device)
        notmnist_probs, _ = get_predictions_and_targets(ensemble_model, notmnist_loader, device)

        mnist_entropy = compute_predictive_entropy(mnist_probs).cpu()
        notmnist_entropy = compute_predictive_entropy(notmnist_probs).cpu()

        experiment_results[("ensemble", M)] = {
            "mnist_entropy": mnist_entropy,
            "notmnist_entropy": notmnist_entropy,
            "mnist_mean": mnist_entropy.mean().item(),
            "notmnist_mean": notmnist_entropy.mean().item(),
        }

    # MC Dropout
    mc_model = load_saved_model(mc_dropout_path, device)

    for M in ensemble_sizes:
        print(f"Running MC Dropout M={M}")
        mnist_probs_mc, _ = get_predictions_and_targets_mc_dropout(
            mc_model, mnist_test_loader, device, M
        )
        notmnist_probs_mc, _ = get_predictions_and_targets_mc_dropout(
            mc_model, notmnist_loader, device, M
        )

        mnist_entropy_mc = compute_predictive_entropy(mnist_probs_mc).cpu()
        notmnist_entropy_mc = compute_predictive_entropy(notmnist_probs_mc).cpu()

        experiment_results[("mc_dropout", M)] = {
            "mnist_entropy": mnist_entropy_mc,
            "notmnist_entropy": notmnist_entropy_mc,
            "mnist_mean": mnist_entropy_mc.mean().item(),
            "notmnist_mean": notmnist_entropy_mc.mean().item(),
        }

    return experiment_results