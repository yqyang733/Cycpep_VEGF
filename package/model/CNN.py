from torch import nn

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()

        self.conv1=nn.Sequential(
            nn.Conv1d(              
                in_channels=1, 
                out_channels=8,
                kernel_size=5,
                stride=1, 
                padding=2,  
            ), 
            nn.ReLU(),  
            nn.MaxPool1d(kernel_size=2),   
        )
 
        self.conv2=nn.Sequential(
            nn.Conv2d(            
                in_channels=8,    
                out_channels=16,  
                kernel_size=5,    
                stride=1,     
                padding=2,       
            ),               
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  
        )
 
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)   
        x=x.view(x.size(0),-1) 
        output=self.fc(x)
        return output
    
    def fc(self, x):
        
        out = nn.Linear(x.shape()[1], 1)
        out = out(x)

        return out
        