#Training loop
import torch
import wandb

def training_loop(model, train_loader, optimizer, criterion, num_of_epochs, wandb_enabled=False, val_loader=None, model_save_path="models/model.pt"):
    for epoch in range(num_of_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y  in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        average_train_loss = train_loss / len(train_loader)

        if val_loader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y  in val_loader:
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