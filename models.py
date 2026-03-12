# Classification and regression models
import torch

class ClassificationModel(torch.nn.Module):
    def __init__(self, input_dims=784, hidden_dims=128, output_dims=10, num_of_hidden_layers=3):
        
        super(ClassificationModel, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.num_of_hidden_layers = num_of_hidden_layers

        assert self.num_of_hidden_layers >= 1 and self.num_of_hidden_layers <= 3, "Only 1 to 3 hidden layers are supported"

        self.linear1 = torch.nn.Linear(input_dims, hidden_dims)
        self.linear2 = torch.nn.Linear(hidden_dims, hidden_dims)
        self.linear3 = torch.nn.Linear(hidden_dims, hidden_dims)
        self.linear_final = torch.nn.Linear(hidden_dims, output_dims)

        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.relu3 = torch.nn.ReLU()

        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_dims)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_dims)
        self.batch_norm3 = torch.nn.BatchNorm1d(hidden_dims)

    def forward(self, x):
        x = x.view(-1, self.input_dims)  

        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        if self.num_of_hidden_layers >= 2:
            x = self.linear2(x)
            x = self.batch_norm2(x)
            x = self.relu2(x)

        if self.num_of_hidden_layers == 3:
            x = self.linear3(x)
            x = self.batch_norm3(x)
            x = self.relu3(x)

        x = self.linear_final(x)

        return x
    
class ClassificationEnsemble(torch.nn.Module):
    def __init__(self, models):
        super(ClassificationEnsemble, self).__init__()
        
        self.models = models

    def forward(self, x):
        outputs = []
        
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        avgerage_output = torch.mean(torch.stack(outputs), dim=0)
        return avgerage_output