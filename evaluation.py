# Evaluation metrics and functions
import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

#CLASSIFICATION

def get_predictions_and_targets(model, test_loader, device):
    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for batch_x, batch_y  in test_loader:
            output = model(batch_x.to(device))

            test_targets.append(batch_y.cpu())
            test_predictions.append(output.cpu())

    test_targets = torch.cat(test_targets, dim=0)
    test_predictions = torch.cat(test_predictions, dim=0)

    return test_predictions, test_targets


def get_predictions_and_targets_mc_dropout(model, test_loader, device, num_samples):
    test_predictions = []
    test_targets = []

    for batch_x, batch_y  in test_loader:
        output = model.average_of_multiple_forward_passes(batch_x.to(device), num_samples)

        test_targets.append(batch_y.cpu())
        test_predictions.append(output.cpu())

    test_targets = torch.cat(test_targets, dim=0)
    test_predictions = torch.cat(test_predictions, dim=0)

    return test_predictions, test_targets


def compute_negative_log_likelihood(probabilities, targets):
    log_probabilities = torch.log(probabilities + 1e-9)
    
    return F.nll_loss(log_probabilities, targets).item()

def compute_classification_error(probabilities, targets):
    num_samples = probabilities.shape[0]

    predicted_classes = torch.argmax(probabilities, dim=1)
    num_incorrect_predictions = (predicted_classes != targets).sum()

    classification_error = num_incorrect_predictions / num_samples
    
    return classification_error.item()

def compute_brier_score(probabilities, targets):
    num_classes = probabilities.shape[1]
    one_hot_encoded_targets = F.one_hot(targets, num_classes=num_classes).to(torch.float32)

    brier_score = torch.mean((probabilities - one_hot_encoded_targets) ** 2)
    return brier_score.item()

#REGRESSION

def nll_criterion_regression(predictions, targets):
    means, variances = predictions
    return torch.mean( torch.log(variances)/2 + (targets - means)**2 / (2 * variances) ) 


def get_predictions_and_targets_rescaled(model, test_loader, output_mean, output_std, device):
    pass # TODO