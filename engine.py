import torch.optim as optim
from model.DSTFGCN import *
import util

criterion = nn.SmoothL1Loss()
class trainer():
    def __init__(self, config, scaler, adjacency_matrix, device):
        self.model = DSTFGCN(device, config, adjacency_matrix, config['hidden_dimension'] * 8)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = config['learning_rate'], weight_decay = config['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                    milestones = config['lr_decay_milestones'], gamma = config['learning_rate_decay'])
        self.loss = util.masked_huber
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, time_i):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, time_i)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val, time_i):
        self.model.eval()
        output = self.model(input, time_i)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
