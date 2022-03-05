import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils

import os
import copy
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=16)
parser.add_argument('--data_dir', type=str, default='.data')
parser.add_argument('--n_megabatches', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_epochs', type=int, default=100)
args = parser.parse_args()

# "real" amount of megabatches
MAX_MB = 500
assert MAX_MB % args.n_megabatches == 0, 'wait time needs to be an integer'
wait_time = MAX_MB // args.n_megabatches

ds_transforms = transforms.Compose([transforms.ToTensor(), lambda x : x.flatten(0)])
train_ds = datasets.MNIST(args.data_dir, download=True,
                        train=True, transform=ds_transforms)
test_ds  = datasets.MNIST(args.data_dir, download=True,
                        train=False, transform=ds_transforms)

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=True)


# split training dataset into multiple chunks
np.random.seed(432)
idx = np.arange(len(train_ds))
idx = np.random.permutation(idx)
chunks = np.array_split(idx, args.n_megabatches)

# now split each chunk into training and validation
tr_chunk, val_chunk = [], []
for chunk in chunks: 
    S = int(chunk.shape[0] * 0.9)
    tr_chunk  += [chunk[:S]]
    val_chunk += [chunk[S:]]

# Helper Methods 
def get_acc(model, loader):
    correct, total = 0, 0
    training = model.training
    model.eval()

    with torch.no_grad():
        for tit, (x,y) in enumerate(val_loader): 
            correct += model(x).argmax(1).eq(y).sum().item()
            total   += x.size(0)

    model.train(training)
    return correct / total

tr_chunks  = np.stack(tr_chunk)
val_chunks = np.stack(val_chunk)

# create model
model = nn.Sequential(
        nn.Linear(784, args.dim), 
        nn.ReLU(), 
        nn.Linear(args.dim, args.dim), 
        nn.ReLU(),
        nn.Linear(args.dim, args.dim), 
        nn.ReLU(),
        nn.Linear(args.dim, 10)
)

opt = torch.optim.Adadelta(model.parameters())

# Useful Placeholders 
test_accs, unique_accs = [], []
data_idx  = 0
last_test_acc  = 0.1 # random chance

# --- START TRAINING
for mb in range(MAX_MB):
    
    # we have acc. enough data, train!
    if (mb + 1) % wait_time == 0: 
        tr_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_ds, tr_chunks[data_idx]), 
                batch_size=args.batch_size, 
                shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_ds, val_chunks[data_idx]), 
                batch_size=args.batch_size, 
                shuffle=True
        )

        data_idx += 1
        
        best_acc   = 0
        best_model = copy.deepcopy(model)
        
        for epoch in range(args.n_epochs):
            
            model.train()
            for it, (x,y) in enumerate(tr_loader):

                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()

            val_acc = get_acc(model, val_loader)
            
            if val_acc > best_acc: 
                best_model = copy.deepcopy(model)
                best_acc   = val_acc
                print(f'mb {mb} / {MAX_MB}\tepoch {epoch}\t acc {best_acc:.4f}', end='\r')

        # Test Accuracy : 
        model.eval()
        last_test_acc = get_acc(model, test_loader)
        unique_accs += [last_test_acc]
        
        # reset from the best model
        model.load_state_dict(best_model.state_dict())
   
        if mb > 0: 
            print(f'\nafter MB {mb}, ', [int(x * 100) for x in unique_accs], 
                  'Avg. Err rate : ', int(sum(test_accs) / len(test_accs) * 100))

    # --- MB over 
    # store the test accuracy
    test_accs += [last_test_acc]


print(data_idx, args.n_megabatches)
import pdb; pdb.set_trace()
xx = 1


