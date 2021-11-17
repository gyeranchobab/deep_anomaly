import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pickle
import os

class BaseModel(nn.Module):
    def __init__(self, model_name='base_model'):
        super().__init__()
        self.model_name = model_name
        self.e = 0
        self.logs = {'train_loss':[], 'eval_loss':[]}
        
    def train_model(self, train_loader, test_loader, epoch, optimizer, report_intv=1, **kwargs):
        best_eval = float('inf')
        
        while True:
            if self.e >= epoch:
                break
            ts=time.time()
            
            self.train()
            train_loss, train_score = self.epoch_step(train_loader, optimizer=optimizer, **kwargs)
            self.logs['train_loss'].append(train_loss)
            
            self.eval()
            with torch.no_grad():
                eval_loss, eval_score = self.epoch_step(test_loader, **kwargs)
            self.logs['eval_loss'].append(eval_loss)
                
            if self.e%report_intv == 0:
                self.report(ts, train_loss, eval_loss, train_score, eval_score)
            self.e += 1
    
    def epoch_step(self, dataloader, optimizer=None, **kwargs):
        avg_loss = 0.0
        avg_score = 0.0
        n=0
        for batch, label, mask in dataloader:
            batch=batch.cuda().to(dtype=torch.float)
            label=label.cuda()#.to(dtype=torch.long)
            if label.dtype == torch.int32:
                label = label.to(dtype=torch.long)
            elif label.dtype == torch.double:
                label = label.to(dtype=torch.float)
            mask=mask.cuda().to(dtype=torch.bool)
            
            masked_label = label.clone()
            masked_label[~mask] = -1
            
            loss, score = self.batch_step(batch, masked_label, mask, optimizer=optimizer, **kwargs)
            
            avg_loss = (n*avg_loss + loss*mask.sum()) / (n+mask.sum()+1e-10)
            avg_score = (n*avg_score + score*mask.sum()) / (n+mask.sum()+1e-10)
            n = n+mask.sum()#batch.shape[0]
        return avg_loss, avg_score
    
    def batch_step(self, batch, label, mask, optimizer=None):
        batch = self.preprocess(batch, label, mask)
        pred = self.forward(batch, label, mask)
        loss = self.criterion(pred, batch, label, mask)
        score = self.compute_score(pred, batch, label, mask)
        
        if self.training and (loss is not None):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        if loss is None:
            loss = 0.0
        elif isinstance(loss, torch.Tensor):
            loss = loss.item()
        if score is None:
            score = 0.0
        elif isinstance(score, torch.Tensor):
            score = score.item()
        
        return loss, score
            
    def preprocess(self, batch=None, label=None, mask=None):
        return batch
        
    def criterion(self, pred=None, batch=None, label=None, mask=None):
        raise NotImplementedError
    
    def compute_score(self, pred=None, batch=None, label=None, mask=None):
        return None
    
    def forward(self, batch, label, mask):
        raise NotImplementedError
    
    def report(self, ts, train_loss, eval_loss, train_score, eval_score):
        if not os.path.exists('./saved_models'):
            os.mkdir('./saved_models')
        if not os.path.exists('./train_logs'):
            os.mkdir('./train_logs')

        print('(%.2fs) [Epoch %d]'%(time.time()-ts, self.e+1))
        print('\tTrain Loss : %.5g\tTrain Score : %.5g'%(train_loss, train_score))
        print('\tEval Loss : %.5g\tEval Score : %.5g'%(eval_loss, eval_score))
        torch.save(self.state_dict(), './saved_models/%s_e%d.pth'%(self.model_name, self.e+1))
        with open('./train_logs/%s.bin'%(self.model_name), 'wb') as f_log:
            pickle.dump(self.logs, f_log)
            
    def predict(self, dataloader, soft_pred=False):
        preds = []
        self.eval()
        for batch, label, mask in dataloader:
            batch=batch.cuda().to(dtype=torch.float)
            label=label.cuda()#.to(dtype=torch.long)
            if label.dtype == torch.int32:
                label = label.to(dtype=torch.long)
            elif label.dtype == torch.double:
                label = label.to(dtype=torch.float)
            mask=mask.cuda().to(dtype=torch.bool)
            
            masked_label = label.clone()
            masked_label[~mask] = -1
            
            batch = self.preprocess(batch, label, mask)
            with torch.no_grad():
                pred = self.forward(batch, label, mask)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
            if soft_pred:
                pred = F.softmax(pred, dim=-1)
            else:
                pred = torch.argmax(pred, dim=1)
            preds.append(pred.cpu())
        preds = torch.cat(preds, 0)
        
        return preds