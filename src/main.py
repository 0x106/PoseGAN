import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        self.nz = 3
        self.nhl = 4
        self.nh = [500, 200, 200, 500]
        self.no = 1

        self.input = nn.Linear(self.nz, self.nh[0])

        self.fc = []
        for i in range(self.nhl-1):
            self.fc.append(nn.Linear(self.nh[i], self.nh[i+1]))

        self.output = nn.Linear(self.nh[self.nhl-1], self.no)

        self.drop = nn.Dropout(p=0.0)

        self.bn = nn.BatchNorm1d(self.nh[0])

    def forward(self, _input):
        x = self.drop(F.relu(self.input((_input))))

        x = self.bn(x)

        for i in range(self.nhl-1):
            x = self.drop(F.relu(self.fc[i]((x))))
            # x = self.bn(x)

        output = F.sigmoid(self.output(x))

        return output

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()

        self.nz = 5
        self.nhl = 4
        self.nh = [2000, 1000, 1000, 2000]
        self.no = 3

        self.input = nn.Linear(self.nz, self.nh[0])

        self.fc = []
        for i in range(self.nhl-1):
            self.fc.append(nn.Linear(self.nh[i], self.nh[i+1]))

        self.output = nn.Linear(self.nh[self.nhl-1], self.no)

        self.drop = nn.Dropout(p=0.0)

        self.bn = nn.BatchNorm1d(self.nh[0])

    def forward(self, _input):

        x = self.drop(F.relu(self.input((_input))))
        x = self.bn(x)

        for i in range(self.nhl-1):
            x = self.drop(F.relu(self.fc[i]((x))))
            # x = self.bn(x)

        output = self.output(x)

        return output

def gen_data(N):

    n, p = 10, .5  # number of trials, probability of each trial
    # s = np.random.binomial(n, p, 1000)
    # s = np.random.beta(0.1, 0.9, size=5000)
    s = np.random.poisson(2, 5000) + np.random.rand(5000)
    r = np.random.poisson(20, 5000) + np.random.rand(5000)

    # s = np.random.normal(0., 1., N)#3*N/4)
    # r = np.random.normal(10., 1., N/4)

    # return s[:, np.newaxis]

    return shuffle(np.concatenate((s, r)))[:, np.newaxis]

def add_noise(_data, _sf):

    size = _data.shape
    tmp = []

    for i in range(_sf):
        noise = (np.random.rand(size[0], size[1]) - 0.5) * 0.25
        tmp.append(_data[:size[0],:]+noise)
    
    for i in range(_sf):
        _data = np.concatenate((_data, tmp[i]))

    return _data

def numpy_to_tensor(_data):
    output_tensor = torch.FloatTensor(_data.shape[0], _data.shape[1])
    for i in range(_data.shape[0]):
        for k in range(_data.shape[1]):
            output_tensor[i][k] = _data[i,k]
    return output_tensor

def shuffle(_data):
    np.random.shuffle(_data)
    return _data

def readData(_files, _num_frames):
    output = '/helix/GAN/data/temp.amc'
    
    if len(_files) == 0:
        _files = ['/helix/GAN/data/06_01.amc']

    print '\t-', _files[0]

    data, _num_frames = [], -1
    for fidx, f in enumerate(_files):
        file = open(f)
        out = open(output, 'w')
        lines = [line.split() for line in file]
        for i in range(3, len(lines)):
            if len(lines[i]) == 1:
                for k in range(1,28):
                    for j in range(1, len(lines[i+k])):
                        out.write(lines[i+k][j])
                        out.write(' ')
                out.write('\n')
        out.close()
        file = open(output)
        lines = [[float(x) for x in line.split()] for line in file]

        _num_frames = len(lines)

        print 'reading data:', _num_frames, 'of (', len(lines), ') frames from', f
        data.append(np.zeros((_num_frames, len(lines[0]))))
        index = np.linspace(0, len(lines)-1, num=_num_frames).astype(int)
        for i in range(_num_frames):
            for k in range(len(lines[0])):
                data[fidx][i,k] = lines[index[i]][k]

    #     x, y, z = data[fidx][:,0], data[fidx][:,1],data[fidx][:,2]
    #     ax.plot(x, y, z, '+')
    # plt.show()

    output = data[0]

    for i in range(len(data)-1):
        output = np.concatenate((output, data[i+1]))

    return output[:, :3]


num_examples = 500
# data = gen_data(num_examples)

# dcount, dbins, dignored = plt.hist(data, 300, normed=True)
# plt.show()

# print 'Getting AMC file for subject:'

files = []
# files.append('/helix/GAN/data/06_01.amc')
# files.append('/helix/GAN/data/07_02.amc')
files.append('/helix/GAN/data/08_02.amc')
# files.append('/helix/GAN/data/08_04.amc')
# files.append('/helix/GAN/data/12_01.amc')

mocap = readData(files, -1)#[:num_examples, :]

print 'Data size (1):', mocap.shape

# x, y, z = mocap[:,0], mocap[:,1],mocap[:,2]
# ax.plot(x, y, z, '+')

mocap = add_noise(mocap,6)
print 'Data size (2):', mocap.shape
# x, y, z = mocap[:,0], mocap[:,1],mocap[:,2]
# ax.plot(x, y, z, '+')
# plt.show()

num_examples = min(2000, mocap.shape[0])
mocap = shuffle(mocap)[:num_examples,:]

print 'Data size (3):', mocap.shape
# count, bins, ignored = plt.hist(mocap, 60, normed=True)

# plt.show()

netG = _netG()
netD = _netD()

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.01)
optimizerG = optim.Adam(netG.parameters(), lr = 0.01)

epochs, batch_size = 100, 400
num_iters = num_examples / batch_size
real_label, fake_label = 1., 0.

# # data = numpy_to_tensor(shuffle(data))
__data = numpy_to_tensor(mocap)

D_input = Variable(torch.FloatTensor(batch_size, 3)) # input to D
label = Variable(torch.FloatTensor(batch_size, 1)) # labels output from D
G_input = Variable(torch.FloatTensor(batch_size, 5)) # input to G

D_error_list, G_error_list, dx_list = [], [], []
exit = False

# print __data.size()

for e in range(epochs):

    idx = 0
    for i in range(num_iters):

        # ===== TRAIN DISCRIMINATOR ====== #
        # ----- train with real data ----- # 
        netD.zero_grad()

        # train D on a real example (i.e one sample from 'data')
        # print D_input.size(), __data[idx:idx+batch_size, :].size()    
        D_input.data.copy_(__data[idx:idx+batch_size, :])
        label.data.fill_(real_label)

        D_real_output = netD(D_input)
        D_real_error  = criterion(D_real_output, label)

        D_real_error.backward()
        D_x = D_real_output.data.mean()

        # ----- train with fake data ----- #
        G_input.data.normal_(0., 1.)
        label.data.fill_(fake_label)

        # pass random noise through G
        G_output = netG(G_input).detach()

        # pass G output through D
        D_fake_output = netD(G_output)

        # compute the D error
        D_fake_error = criterion(D_fake_output, label)
        D_fake_error.backward()

        D_G_z1 = D_fake_output.data.mean()

        # real error is the loss at D for samples that
        # come from the training distribution. The fake
        # error is the loss at D for samples produced by G.
        # Therefore when the D_error is LOW it means that D
        # correctly identifies which distribution its input
        # is produced by. We want this to be low, as we 
        # want D to be as powerful as possible
        D_error = D_fake_error + D_real_error
        D_error_list.append(D_error.data[0])
        dx_list.append(D_x)

        if D_error.data[0] < 0.1:
            exit = True

        optimizerD.step()

        # ======== TRAIN GENERATOR ======= #

        for k in range(2):
            netG.zero_grad()

            G_input.data.normal_(0., 1.)
            label.data.fill_(real_label)

            G_output = netG(G_input)
            D_output = netD(G_output)

            # If D says this is real then G has done
            # it's job well. The loss "at G" is therefore
            # low when G performs well.
            G_error = criterion(D_output, label)

            G_error_list.append(G_error.data[0])

            G_error.backward()

            D_G_z2 = D_output.data.mean()

            optimizerG.step()

        # A low loss at D and a high loss at G means that
        # we aren't producing very realistic samples

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (e, epochs, i, num_iters,
                 D_error.data[0], G_error.data[0], D_x, D_G_z1, D_G_z2))

        # -------------- end ------------- #
        idx += batch_size

        if exit:
            break

    if exit:
        break

# # num_evals = 100
netG.eval()
eval_data = np.zeros((num_examples/10, 3))

for i in range(num_examples/10):
    G_input.data.normal_(0., 1.)
    G_output = netG(G_input)
    eval_data[i, 0] = G_output.data[0][0]
    eval_data[i, 1] = G_output.data[0][1]
    eval_data[i, 2] = G_output.data[0][2]

x, y, z = mocap[:,0], mocap[:,1],mocap[:,2]
x_, y_, z_ = eval_data[:,1], eval_data[:,2], eval_data[:,0]

# plt.subplot(211)
ax.plot(x, y, z, '+')
ax.plot(x_, y_, z_, '+')

# plt.show()


# dcount, dbins, dignored = plt.hist(data, 30, normed=True)
# ecount, ebins, eignored = plt.hist(eval_data, 30, normed=True)

# plt.subplot(212)
# plt.plot(D_error_list)
# plt.plot(G_error_list)

plt.show()


# 