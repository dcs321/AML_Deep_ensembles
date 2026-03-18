#Training loop
import torch
import wandb

from augmentation import generate_adversarially_augmented_samples, generate_randomly_augmented_samples

def training_loop(model, train_loader, optimizer, criterion, num_of_epochs, device, wandb_enabled=False, val_loader=None, model_save_path="models/model.pt", augment=False, type_of_augmentation="random", augmentation_eps = 0.01):
    
    if augment and type_of_augmentation not in ["random", "adversarial"]:
        raise ValueError("Invalid augmentation type. It should be random or adversarial.")
    
    for epoch in range(num_of_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y  in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            if augment:
                if type_of_augmentation == "adversarial":
                    batch_augmented_x = generate_adversarially_augmented_samples(model, criterion, batch_x, batch_y, eps=augmentation_eps)
                elif type_of_augmentation == "random":
                    batch_augmented_x = generate_randomly_augmented_samples(batch_x, eps=augmentation_eps)
                optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            if augment:
                output_augmented = model(batch_augmented_x)
                loss_augmented = criterion(output_augmented, batch_y)
                loss = loss + loss_augmented
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        average_train_loss = train_loss / len(train_loader)

        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y  in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
            average_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} | Train Loss: {average_train_loss} | Val Loss: {average_val_loss}")
            if wandb_enabled:
                wandb.log({
                    'Epoch': epoch + 1,
                    'Train Loss': average_train_loss,
                    'Val Loss': average_val_loss
                })
        else:
            print(f"Epoch {epoch+1} | Train Loss: {average_train_loss}")
            if wandb_enabled:
                wandb.log({
                    'Epoch': epoch + 1,
                    'Train Loss': average_train_loss,
                })
    torch.save(model, model_save_path)
    if wandb_enabled:
        wandb.save(model_save_path)
    
    return model