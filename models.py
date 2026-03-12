# Classification and regression models
import torch
import torch.nn.functional as F

#CLASSIFICATION

class ClassificationModel(torch.nn.Module):
    def __init__(self, input_dims=784, hidden_dims=200, output_dims=10, num_of_hidden_layers=3):
        
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
            probabilities = F.softmax(output, dim=1)
            outputs.append(probabilities)
        
        avgerage_output = torch.mean(torch.stack(outputs), dim=0)
        return avgerage_output
    
class ClassificationMCDropoutModel(ClassificationModel):
    def __init__(self, input_dims=784, hidden_dims=200, output_dims=10, num_of_hidden_layers=3, dropout_rate=0.1):
        super(ClassificationMCDropoutModel, self).__init__(input_dims, hidden_dims, output_dims, num_of_hidden_layers)
        
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.dropout3 = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(-1, self.input_dims)  

        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        if self.num_of_hidden_layers >= 2:
            x = self.linear2(x)
            x = self.batch_norm2(x)
            x = self.relu2(x)
            x = self.dropout2(x)

        if self.num_of_hidden_layers == 3:
            x = self.linear3(x)
            x = self.batch_norm3(x)
            x = self.relu3(x)
            x = self.dropout3(x)

        x = self.linear_final(x)

        return x
    
    def average_of_multiple_forward_passes(self, x, num_passes):
        self.eval() 
        for i,module in enumerate(self.modules()): # Dropout shpuld be enabled, but BatchNorm not
            if isinstance(module, torch.nn.Dropout):
                module.train()

        outputs = []
        with torch.no_grad():
            for i in range(num_passes):
                output = self.forward(x)
                probabilities = F.softmax(output, dim=1)
                outputs.append(probabilities)
        average_output = torch.mean(torch.stack(outputs), dim=0)
        return average_output
    
#REGRESSION

class RegressionModel(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims=50, output_dims=1, num_of_hidden_layers=1):
        super(RegressionModel, self).__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        self.num_of_hidden_layers = num_of_hidden_layers

        assert self.num_of_hidden_layers == 1, "Only 1 hidden layers are supported"
        assert output_dims == 1, "Only 1-dimension output is supported for regression"

        self.linear1 = torch.nn.Linear(input_dims, hidden_dims)
        self.linear_mean = torch.nn.Linear(hidden_dims, output_dims)
        self.linear_variance = torch.nn.Linear(hidden_dims, output_dims)

        self.relu1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)

        x_mean = self.linear_mean(x)
        x_var = self.linear_variance(x)

        x_var = F.softplus(x_var) + 1e-6

        return x_mean, x_var


class RegressionEnsemble(torch.nn.Module):
    def __init__(self, models):
        super(RegressionEnsemble, self).__init__()
        
        self.models = models

    def forward(self, x):
        means = []
        variances = []
        
        for model in self.models:
            mean, variance = model(x)
            means.append(mean)
            variances.append(variance)
        
        means = torch.stack(means)
        variances = torch.stack(variances)

        unified_mean = torch.mean(means, dim=0)
        unified_variance = torch.mean(variances + means**2, dim=0) - unified_mean**2
        
        return unified_mean, unified_variance