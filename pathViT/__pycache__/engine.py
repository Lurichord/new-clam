import os
import numpy as np
from tqdm import tqdm

import time

from sksurv.metrics import concordance_index_censored

import torch.optim
import torch.nn.parallel


class StudentEngine(object):
    def __init__(self, args, results_dir, fold, 
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        self.device = device
        
        # tensorboard，准备开启日志记录，暂时不管
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
            
        # 状态追踪
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None
        

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        # if torch.cuda.is_available():
        model = model.to(self.device)

        for epoch in range(self.args.num_epoch):
            
            print(f"\n[Epoch: {epoch}]")
            self.epoch = epoch
            
            time.sleep(5)
            
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer)
            
            # evaluate on validation set
            c_index = self.validate(val_loader, model, criterion)
            
            # remember best c-index and save checkpoint
            is_best = c_index > self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score})
            print(' *** best c-index={:.4f} at epoch {}'.format(self.best_score, self.best_epoch))
            
            
        return self.best_score, self.best_epoch

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        train_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Train Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(dataloader):
            
            data_WSI = data_WSI.to(self.device)
            # data_omic1 = data_omic1.type(torch.FloatTensor).to(self.device)
            # data_omic2 = data_omic2.type(torch.FloatTensor).to(self.device)
            # data_omic3 = data_omic3.type(torch.FloatTensor).to(self.device)
            # data_omic4 = data_omic4.type(torch.FloatTensor).to(self.device)
            # data_omic5 = data_omic5.type(torch.FloatTensor).to(self.device)
            # data_omic6 = data_omic6.type(torch.FloatTensor).to(self.device)
            
            label = label.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)

            _, hazards, S= model(x_path=data_WSI)

            # survival loss 
            loss = criterion(hazards=hazards, S=S, Y=label, c=c)    
            loss_value = loss.item()

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            
            # # check for NaN
            # if np.isnan(risk).any():
            #     print(f"Warning: NaN detected in risk scores at batch {batch_idx}")
            #     print(f"Hazards: {hazards}")
            #     print(f"S: {S}")
            #     # Optional: Handle NaN, e.g., replace with mean or 0, or skip
            #     # risk = np.nan_to_num(risk)
                
               
                
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time  #event_time只用来测c-index，不用于计算loss
            
            train_loss += loss_value
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # 删除不需要的变量
            del data_WSI, loss, hazards, S
            torch.cuda.empty_cache()


                
            
        # calculate loss and error for epoch
        train_loss /= len(dataloader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                             all_event_times, 
                                             all_risk_scores, 
                                             tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(train_loss, c_index))

        if self.writer:
            self.writer.add_scalar('train/loss', train_loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)

    def validate(self, data_loader, model, criterion):
        model.eval()
        val_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Test Epoch: {}'.format(self.epoch))
        
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(dataloader):
            data_WSI = data_WSI.to(self.device)
            # data_omic1 = data_omic1.type(torch.FloatTensor).to(self.device)
            # data_omic2 = data_omic2.type(torch.FloatTensor).to(self.device)
            # data_omic3 = data_omic3.type(torch.FloatTensor).to(self.device)
            # data_omic4 = data_omic4.type(torch.FloatTensor).to(self.device)
            # data_omic5 = data_omic5.type(torch.FloatTensor).to(self.device)
            # data_omic6 = data_omic6.type(torch.FloatTensor).to(self.device)
            label = label.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)

            with torch.no_grad():
                _, hazards, S= model(x_path=data_WSI, 
                                        # x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3,
                                        # x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6
                                        )  

            # survival loss 
            loss = criterion(hazards=hazards, S=S, Y=label, c=c)
            loss_value = loss.item()

            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time
            
            val_loss += loss_value
            
            
            # 删除不需要的变量
            del data_WSI, loss, hazards, S
            torch.cuda.empty_cache()

        val_loss /= len(dataloader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                             all_event_times, 
                                             all_risk_scores, 
                                             tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(val_loss, c_index))
        
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, self.epoch)
            self.writer.add_scalar('val/c-index', c_index, self.epoch)
        return c_index



    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
