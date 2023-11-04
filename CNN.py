import torch.nn as nn
import torch.nn.functional as F
import torch

# our class must extend nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
        # This applies Linear transformation to input data.
        self.fc1 = nn.Linear(2, 3)

        # This applies linear transformation to produce output data
        self.fc2 = nn.Linear(3, 2)

    # This must be implemented
    def forward(self, x):
        # Output of the first layer
        x = self.fc1(x)
        # Activation function is Relu. Feel free to experiment with this
        x = F.tanh(x)
        # This produces output
        x = self.fc2(x)
        return x

    # This function takes an input and predicts the class, (0 or 1)
    def predict(self, x):
        # Apply softmax to output
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

