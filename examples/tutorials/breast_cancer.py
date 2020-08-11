import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
import time
import syft as sy

class Parser:
    """Parameters for training"""
    def __init__(self):
        self.epochs = 10
        self.lr = 0.001
        self.test_batch_size = 8
        self.batch_size = 8
        self.log_interval = 10
        self.seed = 1
    
args = Parser()

torch.manual_seed(args.seed)
kwargs = {}
cancer = load_breast_cancer()
X = torch.from_numpy(cancer.data).float()
y = torch.from_numpy(cancer.target).float()
X_test = torch.from_numpy(cancer.data).float()
y_test = torch.from_numpy(cancer.target).float()
# preprocessing
mean = X.mean(0, keepdim=True)
dev = X.std(0, keepdim=True)
mean[:, 3] = 0. # the feature at column 3 is binary,
dev[:, 3] = 1.  # so we don't standardize it
X = (X - mean) / dev
X_test = (X_test - mean) / dev
train = TensorDataset(X, y)
test = TensorDataset(X_test, y_test)
train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test, batch_size=args.test_batch_size, shuffle=True, **kwargs)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30, 32)
        self.fc2 = nn.Linear(32, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = x.view(-1, 30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
optimizer = optim.SGD(model.parameters(), lr=args.lr)

hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
james = sy.VirtualWorker(hook, id="james")

compute_nodes = [bob, alice]


remote_dataset = (list(),list())

train_distributed_dataset = []

for batch_idx, (data,target) in enumerate(train_loader):
    data = data.send(compute_nodes[batch_idx % len(compute_nodes)])
    target = target.send(compute_nodes[batch_idx % len(compute_nodes)])
    remote_dataset[batch_idx % len(compute_nodes)].append((data, target))

def update(data, target, model, optimizer):
    start_modelenc = time.time()
    model.send(data.location)
    end_modelenc = time.time()
    print('model encryption time',end_modelenc - start_modelenc)
    optimizer.zero_grad()
    pred = model(data)
    loss = F.mse_loss(pred.view(-1), target)
    loss.backward()
    optimizer.step()
    return model

bobs_model = Net()
alices_model = Net()

bobs_optimizer = optim.SGD(bobs_model.parameters(), lr=args.lr)
alices_optimizer = optim.SGD(alices_model.parameters(), lr=args.lr)

models = [bobs_model, alices_model]
params = [list(bobs_model.parameters()), list(alices_model.parameters())]
optimizers = [bobs_optimizer, alices_optimizer]

# this is selecting which batch to train on
data_index = 0
# update remote models
# we could iterate this multiple times before proceeding, but we're only iterating once per worker here
for remote_index in range(len(compute_nodes)):
    data, target = remote_dataset[remote_index][data_index]
    start_localupdate=time.time()
    models[remote_index] = update(data, target, models[remote_index], optimizers[remote_index])
    end_localupdate=time.time()
    print('local model update time',end_localupdate - start_localupdate)
# create a list where we'll deposit our encrypted model average
new_params = list()
# iterate through each parameter
for param_i in range(len(params[0])):

    # for each worker
    spdz_params = list()
    for remote_index in range(len(compute_nodes)):
        
        # since SMPC can only work with integers (not floats), we need
        # to use Integers to store decimal information. In other words,
        # we need to use "Fixed Precision" encoding.
        fixed_precision_param = params[remote_index][param_i].fix_precision()
        
        # now we encrypt it on the remote machine. Note that 
        # fixed_precision_param is ALREADY a pointer. Thus, when
        # we call share, it actually encrypts the data that the
        # data is pointing TO. This returns a POINTER to the 
        # MPC secret shared object, which we need to fetch.
        encrypted_param = fixed_precision_param.share(bob, alice, crypto_provider=james)
        
        # now we fetch the pointer to the MPC shared value
        param = encrypted_param.get()
        
        # save the parameter so we can average it with the same parameter
        # from the other workers
        spdz_params.append(param)

    # average params from multiple workers, fetch them to the local machine
    # decrypt and decode (from fixed precision) back into a floating point number
    start_encagg = time.time()
    new_param = (spdz_params[0] + spdz_params[1]).get().float_precision()/2
   

    
    # save the new averaged parameter
    new_params.append(new_param)
    end_encagg = time.time()
print('encrypted aggregation time',end_encagg - start_encagg)
with torch.no_grad():
    for model in params:
        for param in model:
            param *= 0

    for model in models:
        model.get()
    start_globalupdate=time.time()
    for remote_index in range(len(compute_nodes)):
        for param_index in range(len(params[remote_index])):
            params[remote_index][param_index].set_(new_params[param_index])
            end_globalupdate=time.time()
print('global model update time', end_globalupdate - start_localupdate)
print('done')

