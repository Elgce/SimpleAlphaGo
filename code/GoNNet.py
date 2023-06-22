import sys
from util import *
sys.path.append('..')

import time, os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

net_config = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})

class GoNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.obs_size()
        self.action_size = game.action_size()
        self.args = args
        
        super(GoNNet, self).__init__()
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)
        
        self.fc1 = nn.Linear(args.num_channels * (self.board_x - 4) * (self.board_y - 4), 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, self.action_size)
        self.fc4 = nn.Linear(256, 1)
        
        
    def forward(self, s):
        ######################################
        #        YOUR CODE GOES HERE         #
        ###################################### 
        # return None
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))
        # 随机丢弃
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)
        # fc3输出的是目前board下在各个位置的概率p
        # fc4输出的是目前board对应的value值
        pi = self.fc3(s)
        v = self.fc4(s)
        return F.log_softmax(pi, dim=1), torch.tanh(v)
    

class GoNNetWrapper():
    def __init__(self, game):
        self.nnet = GoNNet(game, net_config)
        self.board_x, self.board_y = game.obs_size()
        self.action_size = game.action_size()

        if net_config.cuda:
            self.nnet.cuda()

    def train(self, training_data):
        """
        training_data: list of (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=net_config.lr, weight_decay=0.01)
        # import ipdb; ipdb.set_trace()
        losses = []
        for epoch in range(net_config.epochs):
            print(f'Epoch[{str(epoch + 1)}] ')
            self.nnet.train()
            tlosses = []
            batch_count = int(len(training_data) / net_config.batch_size)

            t = tqdm(range(batch_count), desc='Training NNet')
            for _ in t:
                sample_ids = np.random.randint(len(training_data), size=net_config.batch_size)
                boards, pis, vs = list(zip(*[training_data[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if net_config.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                ######################################
                #        YOUR CODE GOES HERE         #
                ###################################### 
                # Compute loss and backprop
                # loss = (z-v)**2 - (pi^T)logP + c||theta||**2
                out_pi, out_v = self.nnet(boards)
                pi_loss = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                v_loss = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
                loss = pi_loss + v_loss
                tlosses.append(loss.item())
                # import ipdb; ipdb.set_trace()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(np.array(tlosses).mean())
        print(losses)
        return losses


    def predict(self, board):
        """
        predict the policy and value of the given board
        @param board: np array 
        @return (pi, v): a policy vector for the given board, and a float value
        """
        
        tboard = torch.FloatTensor(board.copy().astype(np.float64))
        if net_config.cuda: tboard = tboard.contiguous().cuda()
        tboard = tboard.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(tboard)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if net_config.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
