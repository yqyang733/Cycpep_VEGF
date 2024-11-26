import torch
import torch.nn.functional as F
 
class MLPNet(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(MLPNet,self).__init__()

        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
 
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
