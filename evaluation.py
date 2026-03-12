# Evaluation metrics and functions
import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

def get_predictions_and_targets(model, test_loader):
    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for batch_x, batch_y  in test_loader:
            output = model(batch_x)

            test_targets.append(batch_y)
            test_predictions.append(output)

    test_targets = torch.cat(test_targets, dim=0)
    test_predictions = torch.cat(test_predictions, dim=0)

    return test_predictions, test_targets


def compute_negative_log_likelihood(predictions, targets):
    nll_loss = CrossEntropyLoss()
    return nll_loss(predictions, targets).item()

def compute_classification_error(predictions, targets):
    num_samples = predictions.shape[0]

    predicted_classes = torch.argmax(predictions, dim=1)
    num_incorrect_predictions = (predicted_classes != targets).sum()

    classification_error = num_incorrect_predictions / num_samples
    
    return classification_error.item()

def compute_brier_score(predictions, targets):
    num_classes = predictions.shape[1]

    probabilities = F.softmax(predictions, dim=1)
    one_hot_encoded_targets = F.one_hot(targets, num_classes=num_classes).to(torch.float32)

    brier_score = torch.mean((probabilities - one_hot_encoded_targets) ** 2)
    return brier_score.item()
