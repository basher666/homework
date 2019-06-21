import tensorflow as tf
import torch
import numpy as np
import pickle as pkl
import gym
import torch.optim as optim
from torch.utils import data
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch.nn as nn
from models import policy


humanoid_rollouts = pkl.load(open("expert_data/Humanoid-v2.pkl",'rb'))
obs_mean = humanoid_rollouts['observations'].mean(axis=0)
obs_sqmean = np.square(humanoid_rollouts['observations']).mean(axis=0)
obs_std = np.sqrt(np.maximum(0, obs_sqmean - np.square(obs_mean)))

X = pkl.load(open("expert_data/Humanoid-v2.pkl",'rb'))
Y = X['actions']
Y = np.reshape(Y,(Y.shape[0],-1))
Y = Y.astype(np.float32)
Y = list(Y)
X = X['observations']
X = (X - obs_mean)/(obs_std + 1e-6)
X = X.astype(np.float32)
X = list(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=2000, shuffle=True, random_state=42)

class Dataset(data.Dataset):
    def __init__(self, observations, actions):
        'Initialization'
        self.observations = observations
        self.actions = actions
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.observations)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        obs = self.observations[index]
        act = self.actions[index]
        return obs, act

params = {'batch_size': 100,
          'shuffle': True,
          'num_workers': 4}
training_set = Dataset(X_train, Y_train)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(X_test, Y_test)
validation_generator = data.DataLoader(validation_set, **params)

obs_size, h1_size, h2_size , act_size = X_train[0].shape[0], 100, 50, Y_train[0].shape[0]
model = policy(obs_size, h1_size, h2_size, act_size)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

def test(model):
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for sample, target in validation_generator:
            output = model(sample)
            # sum up batch loss
            test_loss += loss_func(output, target).item()
 
        test_loss /= len(validation_generator.dataset)
        print('\nTest set: Average loss: {:.4f}'.format(test_loss))

for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, sample in enumerate(training_generator, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = sample
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 0:    # print every 10 mini-batches
            print('[%d, %d] loss: %f' %(epoch + 1, i + 1, running_loss/2000))
            running_loss = 0.0

    torch.save(model,'epoch_%d_humanoid.model'%(epoch))
    # testing
    print("running on val set")
    test(model)
    
            
print('Finished Training')

torch.save(model.state_dict(), 'humanoid_policy.model')

obs_stats = dict()
obs_stats['mean'] = obs_mean
obs_stats['std'] = obs_std
with open('obs_stats.pkl','wb') as fp:
    pkl.dump(obs_stats, fp)