import torch
import torch.nn.functional as F
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
    checkpoint_paths = {
        "ensemble": [
            f"models/classification/mnist/epoch20/model_mnist_{i}.pt"
            for i in range(1, 16)
        ],
        "ensemble_random": [
            f"models/classification/mnist/epoch20/model_random_augmentation_mnist_{i}.pt"
            for i in range(1, 16)
        ],
        "ensemble_adversarial": [
            f"models/classification/mnist/epoch20/model_adversarial_augmentation_mnist_{i}.pt"
            for i in range(1, 16)
        ],
        "mc_dropout": "models/classification/mnist/epoch20/model_mnist_mc_dropout.pt",
    }
    return checkpoint_paths

# Figure 3

def compute_entropy_comparison_results(
    mnist_test_loader,
    notmnist_loader,
    device,
    ensemble_sizes=(1, 5, 10),
):
    checkpoint_paths = get_mnist_checkpoint_paths()

    experiment_results = {}

    ensemble_methods = {
      "ensemble": checkpoint_paths["ensemble"],
      "ensemble_random": checkpoint_paths["ensemble_random"],
      "ensemble_adversarial": checkpoint_paths["ensemble_adversarial"],
    }

    # ---------- Ensembles ----------
    for method_name, model_paths in ensemble_methods.items():
        for M in ensemble_sizes:
            print(f"Running {method_name} M={M}")
            ensemble_model = load_saved_ensemble(model_paths[:M], device)

            mnist_probs, _ = get_predictions_and_targets(ensemble_model, mnist_test_loader, device)
            notmnist_probs, _ = get_predictions_and_targets(ensemble_model, notmnist_loader, device)

            mnist_entropy = compute_predictive_entropy(mnist_probs).cpu()
            notmnist_entropy = compute_predictive_entropy(notmnist_probs).cpu()

            experiment_results[(method_name, M)] = {
                "mnist_entropy": mnist_entropy,
                "notmnist_entropy": notmnist_entropy,
                "mnist_mean": mnist_entropy.mean().item(),
                "notmnist_mean": notmnist_entropy.mean().item(),
            }

    # ---------- MC Dropout ----------
    mc_model = load_saved_model(checkpoint_paths["mc_dropout"], device)

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

# Figure 6 

def compute_accuracy_vs_confidence_curve(probabilities, targets, thresholds):

    confidences = probabilities.max(dim=1).values
    predictions = probabilities.argmax(dim=1)

    accuracies = []
    coverages = []

    for tau in thresholds:
        mask = confidences >= tau

        if mask.sum().item() == 0:
            accuracies.append(float("nan"))
            coverages.append(0.0)
        else:
            correct = (predictions[mask] == targets[mask]).float().mean().item()
            accuracies.append(correct * 100.0)   # percentage, like paper
            coverages.append(mask.float().mean().item())

    return {
        "thresholds": list(thresholds),
        "accuracies": accuracies,
        "coverages": coverages,
    }

def compute_confidence_comparison_results(
    mnist_test_loader,
    notmnist_loader,
    device,
    thresholds=None,
    ensemble_size=10,
):

    if thresholds is None:
        thresholds = [round(x, 1) for x in torch.arange(0.0, 1.0, 0.1).tolist()]

    checkpoint_paths = get_mnist_checkpoint_paths()

    experiment_results = {}

    ensemble_methods = {
        "ensemble": checkpoint_paths["ensemble"],
        "ensemble_random": checkpoint_paths["ensemble_random"],
        "ensemble_adversarial": checkpoint_paths["ensemble_adversarial"],
    }

    # ---------- Ensembles ----------
    for method_name, model_paths in ensemble_methods.items():
        print(f"Running {method_name} for Figure 6 with M={ensemble_size}")
        ensemble_model = load_saved_ensemble(model_paths[:ensemble_size], device)

        mnist_probs, mnist_targets = get_predictions_and_targets(ensemble_model, mnist_test_loader, device)
        notmnist_probs, notmnist_targets = get_predictions_and_targets(ensemble_model, notmnist_loader, device)

        mixed_probs = torch.cat([mnist_probs.cpu(), notmnist_probs.cpu()], dim=0)
        mixed_targets = torch.cat(
            [mnist_targets.cpu(), torch.full_like(notmnist_targets.cpu(), -1)],
            dim=0
        )

        experiment_results[method_name] = compute_accuracy_vs_confidence_curve(
            mixed_probs, mixed_targets, thresholds
        )

    # ---------- MC Dropout ----------
    mc_model = load_saved_model(checkpoint_paths["mc_dropout"], device)

    print(f"Running mc_dropout for Figure 6 with M={ensemble_size}")
    mnist_probs_mc, mnist_targets_mc = get_predictions_and_targets_mc_dropout(
        mc_model, mnist_test_loader, device, ensemble_size
    )
    notmnist_probs_mc, notmnist_targets_mc = get_predictions_and_targets_mc_dropout(
        mc_model, notmnist_loader, device, ensemble_size
    )

    mixed_probs_mc = torch.cat([mnist_probs_mc.cpu(), notmnist_probs_mc.cpu()], dim=0)
    mixed_targets_mc = torch.cat(
        [mnist_targets_mc.cpu(), torch.full_like(notmnist_targets_mc.cpu(), -1)],
        dim=0
    )

    experiment_results["mc_dropout"] = compute_accuracy_vs_confidence_curve(
        mixed_probs_mc, mixed_targets_mc, thresholds
    )

    return experiment_results

# qualitative analysis

def collect_ensemble_member_predictions(models, data_loader, device):
    all_images = []
    all_targets = []
    member_outputs = [[] for _ in range(len(models))]

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x_device = batch_x.to(device)

            all_images.append(batch_x.cpu())
            all_targets.append(batch_y.cpu())

            for i, model in enumerate(models):
                logits = model(batch_x_device)
                probs = F.softmax(logits, dim=1).cpu()
                member_outputs[i].append(probs)

    images = torch.cat(all_images, dim=0)
    targets = torch.cat(all_targets, dim=0)

    member_probs = torch.stack(
        [torch.cat(member_outputs[i], dim=0) for i in range(len(models))],
        dim=0
    )  # [M, N, C]

    ensemble_probs = member_probs.mean(dim=0)  # [N, C]
    pred_labels = ensemble_probs.argmax(dim=1)
    confidence = ensemble_probs.max(dim=1).values

    return {
        "images": images,
        "targets": targets,
        "member_probs": member_probs,
        "ensemble_probs": ensemble_probs,
        "pred_labels": pred_labels,
        "confidence": confidence,
    }


def compute_disagreement_scores(member_probs, ensemble_probs):
    member_probs = member_probs.clamp_min(1e-9)
    ensemble_probs = ensemble_probs.clamp_min(1e-9)

    log_member = member_probs.log()
    log_ensemble = ensemble_probs.unsqueeze(0).log()

    kl = (member_probs * (log_member - log_ensemble)).sum(dim=-1)  # [M, N]
    disagreement = kl.mean(dim=0)  # [N]

    return disagreement


def get_extreme_indices(scores, num_each=20):
    sorted_indices = torch.argsort(scores)
    lowest = sorted_indices[:num_each]
    highest = sorted_indices[-num_each:]
    return lowest, highest

def get_class_balanced_extreme_indices(scores, targets, num_per_class=2, descending=False):
    selected_indices = []

    class_labels = torch.unique(targets).tolist()
    class_labels = sorted(int(c) for c in class_labels)

    for c in class_labels:
        class_mask = (targets == c)
        class_indices = torch.where(class_mask)[0]
        class_scores = scores[class_indices]

        order = torch.argsort(class_scores, descending=descending)
        chosen = class_indices[order[:num_per_class]]
        selected_indices.append(chosen)

    return torch.cat(selected_indices, dim=0)

def prepare_qualitative_uncertainty_results(
    data_loader,
    device,
    method_key="ensemble",
    ensemble_size=10,
):
    checkpoint_paths = get_mnist_checkpoint_paths()

    method_to_paths = {
        "ensemble": checkpoint_paths["ensemble"],
        "ensemble_random": checkpoint_paths["ensemble_random"],
        "ensemble_adversarial": checkpoint_paths["ensemble_adversarial"],
    }

    model_paths = method_to_paths[method_key][:ensemble_size]
    models = [load_saved_model(p, device) for p in model_paths]
    # debug
    print(len(models))
    print(type(models[0]))

    results = collect_ensemble_member_predictions(models, data_loader, device)
    disagreement = compute_disagreement_scores(
        results["member_probs"],
        results["ensemble_probs"]
    )

    results["disagreement"] = disagreement
    return results



