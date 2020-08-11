epochs = 10
# We don't use the whole dataset for efficiency purpose, but feel free to increase these numbers
n_train_items = 640
n_test_items = 640
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data

import time
class Arguments():
    def __init__(self):
        self.batch_size = 8
        self.test_batch_size = 8
        self.epochs = epochs
        self.lr = 0.02
        self.seed = 1
        self.log_interval = 1 # Log info at each batch
        self.precision_fractional = 3

args = Arguments()

_ = torch.manual_seed(args.seed)
import syft as sy  # import the Pysyft library
hook = sy.TorchHook(torch)  # hook PyTorch to add extra functionalities like Federated and Encrypted Learning
x_train_features =torch.from_numpy(np.load('x_train_features.npy'))
y_train = torch.from_numpy(np.load('y_train.npy'))

x_test_features = torch.from_numpy(np.load('x_test_features.npy'))
y_test = torch.from_numpy(np.load('y_test.npy'))

#train_loader = (x_train_features, y_train)
#test_loader = (x_test_features,y_test)
my_dataset=data.TensorDataset(x_train_features,y_train)
dataset2 = data.TensorDataset(x_test_features,y_test)

print(x_train_features.shape,y_train.shape,x_test_features.shape,y_test.shape)

# simulation functions
def connect_to_workers(n_workers):
    return [
        sy.VirtualWorker(hook, id=f"worker{i+1}")
        for i in range(n_workers)
    ]
def connect_to_crypto_provider():
    return sy.VirtualWorker(hook, id="crypto_provider")

workers = connect_to_workers(n_workers=2)
crypto_provider = connect_to_crypto_provider()
def get_private_data_loaders(precision_fractional, workers, crypto_provider):
    
    def one_hot_of(index_tensor):
        """
        Transform to one hot tensor
        
        Example:
            [0, 3, 9]
            =>
            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
            
        """
        onehot_tensor = torch.zeros(*index_tensor.shape, 10) # 10 classes for MNIST
        onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
        return onehot_tensor
        
    def secret_share(tensor):
        """
        Transform to fixed precision and secret share a tensor
        """
        return (
            tensor
            .fix_precision(precision_fractional=precision_fractional)
            .share(*workers, crypto_provider=crypto_provider, requires_grad=True)
        )
    
    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(my_dataset,batch_size=args.batch_size)
    
    private_train_loader = [
        (secret_share(data), secret_share(target))
        for i, (data,target) in enumerate(train_loader)
        if i < n_train_items / args.batch_size
    ]
    
    test_loader = torch.utils.data.DataLoader(dataset2,batch_size=args.test_batch_size)
    
    private_test_loader = [
        (secret_share(data), secret_share(target))
        for i, (data,target) in enumerate(test_loader)
        if i < n_test_items / args.test_batch_size
    ]
    
    return private_train_loader, private_test_loader
    
    
private_train_loader, private_test_loader = get_private_data_loaders(
    precision_fractional=args.precision_fractional,
    workers=workers,
    crypto_provider=crypto_provider
)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4732, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 4732)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv = nn.Conv2d(1, 28, kernel_size=3)
        self.pool = nn.AvgPool2d(2)
        self.hidden= nn.Linear(28*13*13, 128)
        #self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(128, 10)
        #self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x) # [batch_size, 28, 26, 26]
        x = F.relu(x)
        x = self.pool(x) # [batch_size, 28, 13, 13]
        x = x.view(args.batch_size, -1) # [batch_size, 28*13*13=4732]
        x = self.hidden(x)
        x = F.relu(x) # [batch_size, 128]
        #x = F.dropout(0.5)
        x = self.out(x) # [batch_size, 10]
        return x

def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(private_train_loader): # <-- now it is a private datase
	
        start_time = time.time()
        print ("Start Training")
        
        optimizer.zero_grad()
        
        output = model(data)
        
        # loss = F.nll_loss(output, target)  <-- not possible here
        batch_size = output.shape[0]
        loss = ((output - target)**2).sum().refresh()/batch_size
        
        loss.backward()
        
        optimizer.step()
        print ("Done first step")
        if batch_idx % args.log_interval == 0:
            loss = loss.get().float_precision()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
                epoch, batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
                100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))

def test(args, model, private_test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in private_test_loader:
            start_time = time.time()
            
            output = model(data)
            pred = output.argmax(dim=1)
            #print(pred)
            #print(target)
            correct += pred.eq(target.argmax(dim=1)).sum()

    correct = correct.get().float_precision()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct.item(), len(private_test_loader)* args.test_batch_size,
        100. * correct.item() / (len(private_test_loader) * args.test_batch_size)))

#model = Net()
model = NeuralNet()
model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)
optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optimizer.fix_precision() 
for epoch in range(1, args.epochs + 1):
    train(args, model, private_train_loader, optimizer, epoch)
    test(args, model, private_test_loader)

