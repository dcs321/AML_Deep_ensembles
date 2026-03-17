import torch

def generate_adversarially_augmented_samples(model, criterion, batch_x, batch_y, eps=0.01):
    batch_x_adv = batch_x.clone().detach()
    batch_x_adv.requires_grad = True

    output = model(batch_x_adv)
    loss = criterion(output, batch_y)
    loss.backward()
    batch_x_augmented = batch_x_adv + eps * torch.sign(batch_x_adv.grad)
    batch_x_augmented = batch_x_augmented.detach()

    return batch_x_augmented

def generate_randomly_augmented_samples(batch_x, eps=0.01):
    batch_x_augmented = batch_x + eps * torch.sign(torch.randn_like(batch_x))
    batch_x_augmented = batch_x_augmented.detach()
    return batch_x_augmented