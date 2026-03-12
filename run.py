# Run training and evaluation
import argparse
import torch

from models import ClassificationModel, ClassificationEnsemble
from data import load_mnist
from training import training_loop
from evaluation import get_predictions_and_targets, compute_negative_log_likelihood, compute_brier_score, compute_classification_error

import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--run_name', type=str, default='ensemble_run', help='Name of the wandb run.')
    parser.add_argument('--classification_or_regresssion', type=str, default='regression', help='Regression or classification.')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use.')
    parser.add_argument('--number_of_models', type=int, default=5, help='Number of models in the ensemble.')
    parser.add_argument('--with_validation_set', action='store_true', default=False, help='Whether to use validation set or not.')
    parser.add_argument('--perform_ensemble_experiment', action='store_true', help='Whether to perform ensemble experiment or not.')

    args = parser.parse_args() 

    if args.wandb:
        wandb.init(name=args.run_name, config=vars(args), save_code=True)
        wandb.run.log_code(".")
 

    OPTIMIZER = torch.optim.Adam
    if args.classification_or_regresssion == "classification":
        LEARNING_RATE = 0.1
        NUMBER_OF_EPOCHS = 10
        CRITERION = torch.nn.CrossEntropyLoss()

        MODEL = ClassificationModel
        if args.dataset == "mnist":
            train_loader, val_loader, test_loader = load_mnist(with_validation_set=args.with_validation_set)
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported for classification.")
    elif args.classification_or_regresssion == "regression":
        raise NotImplementedError("Regression need to be added here.")
    else:
        raise ValueError("Only classification and regression are supported.")
    
    models_in_ensemble = []
    for i in range(args.number_of_models):
        model = MODEL()
        optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)

        print(f"Training ensemble model {i+1}.")
        model = training_loop(model, train_loader, optimizer, CRITERION, NUMBER_OF_EPOCHS, args.wandb, val_loader=val_loader, model_save_path=f"models/model_{args.dataset}_{i+1}.pt")
        
        print(f"Evaluating ensemble model {i+1}.")
        model.eval()
        test_predictions, test_targets = get_predictions_and_targets(model, test_loader)
        test_loss = CRITERION(test_predictions, test_targets)
        print(f"Test loss for model {i+1}: {test_loss.item()}")
        if args.wandb:
            wandb.log({f'Test Loss for model {i+1}': test_loss.item()})
        models_in_ensemble.append(model)

    
    if args.classification_or_regresssion == "classification":

        ensemble_model = ClassificationEnsemble(models_in_ensemble)
        ensemble_model.eval()

        ensemble_predictions, ensemble_targets = get_predictions_and_targets(ensemble_model, test_loader)
        
        ensemble_test_loss = CRITERION(ensemble_predictions, ensemble_targets)
        ensemble_brier_score = compute_brier_score(ensemble_predictions, ensemble_targets)
        ensemble_classification_error = compute_classification_error(ensemble_predictions, ensemble_targets)
        
        print(f"Test loss (NLL) for final ensemble: {ensemble_test_loss.item()}")
        print(f"Brier score for final ensemble: {ensemble_brier_score}")
        print(f"Classification for final ensemble: {ensemble_classification_error}")
        
        if args.wandb:
            wandb.log({f'Final Ensemble NLL': ensemble_test_loss.item()})
            wandb.log({f'Final Ensemble Brier Score': ensemble_brier_score})
            wandb.log({f'Final Ensemble Classification Error': ensemble_classification_error})

        if args.perform_ensemble_experiment:
            print("Ensemble experiment started")
            for i in range(1, args.number_of_models + 1):
                ensemble_model = ClassificationEnsemble(models_in_ensemble[:i])
                ensemble_model.eval()
                
                ensemble_predictions, ensemble_targets = get_predictions_and_targets(ensemble_model, test_loader)
                
                ensemble_nll = compute_negative_log_likelihood(ensemble_predictions, ensemble_targets)
                ensemble_classification_error = compute_classification_error(ensemble_predictions, ensemble_targets)
                ensemble_brier_score = compute_brier_score(ensemble_predictions, ensemble_targets)

                print(f"{i}-Model Ensemble NLL: {ensemble_nll})")
                print(f"{i}-Model Ensemble Classification Error: {ensemble_classification_error}")
                print(f"{i}-Model Ensemble Brier Score: {ensemble_brier_score}")

                if args.wandb:
                    wandb.log({f'Ensemble NLL': ensemble_nll, 'Ensemble Classification Error': ensemble_classification_error, 'Ensemble Brier Score': ensemble_brier_score})

    else:
        pass


if __name__ == "__main__":
    main()