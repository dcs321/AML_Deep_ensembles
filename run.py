# Run training and evaluation
import argparse
import torch
import numpy as np

from models import ClassificationModel, ClassificationEnsemble, ClassificationMCDropoutModel, RegressionModel, RegressionEnsemble
from data import load_mnist, load_boston_housing, load_concrete, load_energy
from training import training_loop
from evaluation import get_predictions_and_targets, compute_negative_log_likelihood, compute_brier_score, compute_classification_error, get_predictions_and_targets_mc_dropout, nll_criterion_regression, get_predictions_and_targets_rescaled, compute_rmse_regression, compute_nll_regression

import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--run_name', type=str, default='ensemble_run', help='Name of the wandb run.')
    parser.add_argument('--classification_or_regression', type=str, default='classification', help='Regression or classification.')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use.')
    parser.add_argument('--number_of_models', type=int, default=5, help='Number of models in the ensemble.')
    parser.add_argument('--number_of_steps', type=int, default=5, help='Number of inference steps just for MC Dropout.')
    parser.add_argument('--with_validation_set', action='store_true', default=False, help='Whether to use validation set or not.')
    parser.add_argument('--perform_ensemble_experiment', action='store_true', help='Whether to perform ensemble experiment or not.')
    parser.add_argument('--perform_mc_dropout_experiment', action='store_true', help='Whether to perform MC Dropout experiment or not.')
    parser.add_argument('--augment', action='store_true', help='Whether to augment the training set or not.')
    parser.add_argument('--type_of_augmentation', type=str, default="random", help='Whether to randomly or adversarially augment the training set.')
    parser.add_argument('--augmentation_eps', type=int, default=0.01, help='Epsilon used during augmentation.')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.wandb:
        wandb.init(name=args.run_name, config=vars(args), save_code=True)
        wandb.run.log_code(".")

    assert (not args.augment) or (args.augment and args.classification_or_regression == "classification"), "Augmentation currently implemented just for classification."

    OPTIMIZER = torch.optim.Adam
    LEARNING_RATE = 0.1

    if args.classification_or_regression == "classification": #CLASSIFICATION
        NUMBER_OF_EPOCHS = 20
        CRITERION = torch.nn.CrossEntropyLoss()

        MODEL = ClassificationModel

        hidden_dims = 200
        num_of_hidden_layers = 3

        if args.dataset == "mnist":
            input_dims = 784
            output_dims = 10
            train_loader, val_loader, test_loader = load_mnist(with_validation_set=args.with_validation_set)
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported for classification.")
    elif args.classification_or_regression == "regression": #REGRESSION
        NUMBER_OF_EPOCHS = 40
        CRITERION = nll_criterion_regression
        LEARNING_RATE = 0.01
        MODEL = RegressionModel
        
        hidden_dims = 50
        output_dims = 1
        num_of_hidden_layers = 1

        num_of_train_test_splits = 20
        train_ratio_in_split = 0.9

        if args.dataset == "boston_housing":
            input_dims = 13
            train_loaders, test_loaders, output_means_for_standardization, output_stds_for_standardization = load_boston_housing(num_of_train_test_splits, train_ratio_in_split)
        elif args.dataset == "concrete":
            input_dims = 8
            train_loaders, test_loaders, output_means_for_standardization, output_stds_for_standardization = load_concrete(num_of_train_test_splits, train_ratio_in_split)
        elif args.dataset == "energy":
            input_dims = 8
            train_loaders, test_loaders, output_means_for_standardization, output_stds_for_standardization = load_energy(num_of_train_test_splits, train_ratio_in_split)
        else:
            raise ValueError(f"Dataset {args.dataset} is not supported for regression.")
            
        assert output_dims == 1, "Only 1-dimension output is supported for regression"
    else:
        raise ValueError("Only classification and regression are supported.")
    
    print("Dataset:", args.dataset)

    if args.classification_or_regression == "classification": #CLASSIFICATION
        models_in_ensemble = []
        for i in range(args.number_of_models):
            model = MODEL(input_dims, hidden_dims, output_dims, num_of_hidden_layers).to(device)
            optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)

            print(f"Training ensemble model {i+1}.")
            if args.augment:
                 model_save_path = f"models/{args.classification_or_regression}/{args.dataset}/model_{args.type_of_augmentation}_augmentation_{args.dataset}_{i+1}.pt"
            else:
                model_save_path = f"models/{args.classification_or_regression}/{args.dataset}/model_{args.dataset}_{i+1}.pt"
            model = training_loop(model, train_loader, optimizer, CRITERION, NUMBER_OF_EPOCHS, device, args.wandb, val_loader=val_loader, model_save_path=model_save_path, augment=args.augment, type_of_augmentation=args.type_of_augmentation, augmentation_eps=args.augmentation_eps)
            
            print(f"Evaluating ensemble model {i+1}.")
            model.eval()
            test_predictions, test_targets = get_predictions_and_targets(model, test_loader, device)
            test_loss = CRITERION(test_predictions, test_targets)
            print(f"Test loss for model {i+1}: {test_loss.item()}")
            if args.wandb:
                wandb.log({f'Test Loss for model {i+1}': test_loss.item()})
            models_in_ensemble.append(model)

        ensemble_model = ClassificationEnsemble(models_in_ensemble)
        ensemble_model.eval()

        ensemble_predictions, ensemble_targets = get_predictions_and_targets(ensemble_model, test_loader, device)
        
        ensemble_test_loss_nll = compute_negative_log_likelihood(ensemble_predictions, ensemble_targets)
        
        print(f"Test loss (NLL) for final ensemble: {ensemble_test_loss_nll}")
        
        if args.wandb:
            wandb.log({f"Test loss (NLL) for final ensemble": ensemble_test_loss_nll})

        if args.perform_ensemble_experiment:
            print("Ensemble experiment started")
            for i in range(1, args.number_of_models + 1):
                ensemble_model = ClassificationEnsemble(models_in_ensemble[:i])
                ensemble_model.eval()
                
                ensemble_predictions, ensemble_targets = get_predictions_and_targets(ensemble_model, test_loader, device)
                
                ensemble_nll = compute_negative_log_likelihood(ensemble_predictions, ensemble_targets)
                ensemble_classification_error = compute_classification_error(ensemble_predictions, ensemble_targets)
                ensemble_brier_score = compute_brier_score(ensemble_predictions, ensemble_targets)

                print(f"{i}-Model Ensemble NLL: {ensemble_nll})")
                print(f"{i}-Model Ensemble Classification Error: {ensemble_classification_error}")
                print(f"{i}-Model Ensemble Brier Score: {ensemble_brier_score}")

                if args.wandb:
                    wandb.log({f'Ensemble NLL': ensemble_nll, 'Ensemble Classification Error': ensemble_classification_error, 'Ensemble Brier Score': ensemble_brier_score})
        if args.perform_mc_dropout_experiment:
            print("MC Dropout training started")
            dropout_rate = 0.1
            dropout_model = ClassificationMCDropoutModel(input_dims, hidden_dims, output_dims, num_of_hidden_layers, dropout_rate).to(device)
            dropout_optimizer = OPTIMIZER(dropout_model.parameters(), lr=LEARNING_RATE)
            dropout_model = training_loop(dropout_model, train_loader, dropout_optimizer, CRITERION, NUMBER_OF_EPOCHS, device, args.wandb, val_loader=val_loader, model_save_path=f"models/{args.classification_or_regression}/{args.dataset}/model_{args.dataset}_mc_dropout.pt")

            dropout_model.eval()
            dropout_test_predictions, dropout_test_targets = get_predictions_and_targets(dropout_model, test_loader, device)
            dropout_test_loss = CRITERION(dropout_test_predictions, dropout_test_targets)
            print(f"Test loss for Dropout model: {dropout_test_loss.item()}")
            if args.wandb:
                wandb.log({f'Test Loss for Dropout model': dropout_test_loss.item()})

            print("MC Dropout experiment started")
            for i in range(1, args.number_of_steps + 1):
                dropout_predictions, dropout_targets = get_predictions_and_targets_mc_dropout(dropout_model, test_loader, device, i)
                
                dropout_nll = compute_negative_log_likelihood(dropout_predictions, dropout_targets)
                dropout_classification_error = compute_classification_error(dropout_predictions, dropout_targets)
                dropout_brier_score = compute_brier_score(dropout_predictions, dropout_targets)

                print(f"{i}-Model Dropout NLL: {dropout_nll})")
                print(f"{i}-Model Dropout Classification Error: {dropout_classification_error}")
                print(f"{i}-Model Dropout Brier Score: {dropout_brier_score}")

                if args.wandb:
                    wandb.log({f'Dropout NLL': dropout_nll, 'Dropout Classification Error': dropout_classification_error, 'Dropout Brier Score': dropout_brier_score})

    elif args.classification_or_regression == "regression": #REGRESSION
        rmse_results = []
        nll_results = []
        for i in range(num_of_train_test_splits):
            models_in_ensemble = []
            for j in range(args.number_of_models):
                model = MODEL(input_dims, hidden_dims, output_dims, num_of_hidden_layers).to(device)
                optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)

                print(f"Training ensemble model {j+1} on split {i+1}.")
                model = training_loop(model, train_loaders[i], optimizer, CRITERION, NUMBER_OF_EPOCHS, device, args.wandb, val_loader=None, model_save_path=f"models/{args.classification_or_regression}/{args.dataset}/model_{args.dataset}_split_{i+1}_model_{j+1}.pt")
                
                model.eval()
                models_in_ensemble.append(model)
        
            ensemble_model = RegressionEnsemble(models_in_ensemble)
            ensemble_model.eval()

            test_predictions, test_targets = get_predictions_and_targets_rescaled(ensemble_model, test_loaders[i], output_means_for_standardization[i], output_stds_for_standardization[i], device)
            #RMSE and NLL computation
            split_rmse = compute_rmse_regression(test_predictions, test_targets)
            split_nll  = compute_nll_regression(test_predictions, test_targets)
            rmse_results.append(split_rmse)
            nll_results.append(split_nll)

            print(f"Split {i+1} | RMSE: {split_rmse:.4f} | NLL: {split_nll:.4f}")
            if args.wandb:
                wandb.log({f'Split {i+1} RMSE': split_rmse, f'Split {i+1} NLL': split_nll})
        
        #compute mean and std of RMSE and NLL across splits for the table
        mean_rmse, std_rmse = np.mean(rmse_results), np.std(rmse_results)
        mean_nll, std_nll = np.mean(nll_results), np.std(nll_results)

        print(f"FINAL RESULTS over {num_of_train_test_splits} splits:")
        print(f"  RMSE : {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"  NLL  : {mean_nll:.4f}  ± {std_nll:.4f}")

        if args.wandb:
            wandb.log({'Mean RMSE': mean_rmse, 'Std RMSE': std_rmse, 'Mean NLL':  mean_nll,  'Std NLL':  std_nll})


if __name__ == "__main__":
    main()
