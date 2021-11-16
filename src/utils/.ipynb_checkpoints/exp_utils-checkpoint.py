import os
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.utils.data_utils import CWRUDataset

def plot_logs(models,log_path='./train_logs', plot_acc=True):
    AX_NAME = ['Train Loss','Test Loss']#
    if plot_acc: 
        AX_NAME += ['Train Acc', 'Test Acc']
    fig=plt.figure(figsize=(10,5*len(AX_NAME)//2))
    for m in models:
        if not os.path.isfile(log_path+'/%s.bin'%m):
            print('No %s'%m)
            continue
        results = pickle.load(open(log_path+'/%s.bin'%m,'rb'))
        for i,r in enumerate(results):
            if i >= len(AX_NAME):
                break
            ax = plt.subplot(len(AX_NAME)//2,2,i+1)
            ax.set_title(AX_NAME[i])
            ax.plot(r, label=m)
            if i<1:
                ax.legend()
    plt.show()
    
def test_model(model, split='test', SNR=None, overwrite=False, test_path='./test_results'):
    if SNR is None:
        SNR = [None,8,4,2,0]
    model.cuda()
    for snr in SNR:
        if not overwrite:
            if os.path.isfile(test_path+'/pred_%s[%s]_%s.npy'%(model.name, snr, split)):
                if os.path.isfile(test_path+'/z_%s[%s]_%s.npy'%(model.name,snr,split)):
                    if os.path.isfile(test_path+'/result_%s[%s]_%s.bin'%(model.name,snr,split)):
                        print('pass %s (%s, SNR=%s)'%(model.name,split,snr))
                        continue
        dataset = CWRUDataset('./data',split,p=1,snr=snr)
        dataloader = DataLoader(dataset,batch_size=32)
        acc,z,pred = model.predict(dataloader)
        pred = pred.numpy()
        
        pred = np.stack([dataset.labels,pred],axis=0)
        np.save(test_path+'/pred_%s[%s]_%s.npy'%(model.name, snr, split), pred)
        z=z.numpy()
        
        np.save(test_path+'/z_%s[%s]_%s.npy'%(model.name,snr,split),z)
        pickle.dump(acc, open(test_path+'/result_%s[%s]_%s.bin'%(model.name,snr,split),'wb'))
        print('(%s) %s (SNR=%s) :\t%s'%(split,model.name,snr,acc))
        
def make_table(model_names, split='test',test_path='./test_results', SNR=None):
    if SNR is None:
        SNR = [None,8,4,2,0]
    for i,m in enumerate(model_names):
        acc_mat = []
        ax = plt.subplot2grid((len(model_names)+2,len(SNR)+1),(i+1,1),colspan=len(SNR),rowspan=1)
        if i==0:
            ax.set_xticks(np.arange(len(SNR)))
            ax.set_xticklabels([x if x is not None else 'None' for x in SNR])
            ax.xaxis.tick_top()
            ax.set_xlabel("SNR")
            ax.xaxis.set_label_position('top')
        else:
            ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(m)
        
        acc_row = []
        for k, snr in enumerate(SNR):
            if os.path.isfile(test_path+'/result_%s[%s]_%s.bin'%(m,snr,split)):
                acc = pickle.load(open(test_path+'/result_%s[%s]_%s.bin'%(m,snr,split), 'rb'))
            else:
                acc = 0.0
            ax.text(k,0,'%.2f'%(100*acc),ha='center',va='center',color='black')
            acc_row.append(acc)
        acc_mat.append(acc_row)
        acc_mat = [[x**2 for x in y] for y in acc_mat]
        ax.imshow(acc_mat,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
    plt.show()
    
def make_confmat(model_name, split='test',test_path='./test_results', snr=None):
    if os.path.isfile(test_path+'/pred_%s[%s]_%s.npy'%(model_name,snr,split)):
        preds = np.load(test_path+'/pred_%s[%s]_%s.npy'%(model_name,snr,split))
        
        y = preds[0]
        preds= preds[1]
        
        matrix = [[len(y[(y==i)&(preds==j)]) for j in range(10)] for i in range(10)]
        fig,ax = plt.subplots()
        im = ax.imshow(matrix, cmap='viridis')
        ax.set_xlabel('Pred')
        ax.set_ylabel('True')
        for i in range(10):
            for j in range(10):
                text = ax.text(j,i,matrix[i][j],ha='center',va='center',color='w')
        fig.tight_layout()
        plt.show()
    else:
        print('no test results')
        
        
def get_tsne(model_name, split='test',test_path='./test_results', snr=None, overwrite=False):
    if (not overwrite) and os.path.isfile(test_path+'/tsne_%s[%s]_%s.npy'%(model_name,snr,split)):
        return np.load(test_path+'/tsne_%s[%s]_%s.npy'%(model_name,snr,split))
    if not os.path.isfile(test_path+'/z_%s[%s]_%s.npy'%(model_name,snr,split)):
        print('no test result')
        return
    z = np.load(test_path+'/z_%s[%s]_%s.npy'%(model_name,snr,split))
    tsne = TSNE()
    z_shrink = tsne.fit_transform(z)
    np.save(test_path+'/tsne_%s[%s]_%s.npy'%(model_name,snr,split), z_shrink)
    return z_shrink

def get_pca(model_name, split='test',test_path='./test_results', snr=None, overwrite=False):
    if (not overwrite) and os.path.isfile(test_path+'/pca_%s[%s]_%s.npy'%(model_name,snr,split)):
        return np.load(test_path+'/pca_%s[%s]_%s.npy'%(model_name,snr,split))
    if not os.path.isfile(test_path+'/z_%s[%s]_%s.npy'%(model_name,snr,split)):
        print('no test result')
        return
    z = np.load(test_path+'/z_%s[%s]_%s.npy'%(model_name,snr,split))
    tsne = PCA()
    z_shrink = tsne.fit_transform(z)
    np.save(test_path+'/pca_%s[%s]_%s.npy'%(model_name,snr,split), z_shrink)
    return z_shrink

    
def draw_plot(ax, f, arg, difference=False, split='test',legend=True):
    if difference:
        pred, dataset = arg
        scatter = ax.scatter(f[:,0],f[:,1],c=dataset.labels==pred,s=1,alpha=0.5,picker=5,cmap='Set1')
        lines, labels = scatter.legend_elements()
        labels = ['Misclassified','Correct']
        if split == 'train':
            scatter = ax.scatter(f[:,0][dataset.mask],f[:,1][dataset.mask],c='blue',s=5,label='Labeled')
            legend2 = scatter.legend_elements()
        if legend:
            legend = ax.legend(lines, labels)
            ax.add_artist(legend)
    else:
        cmap = plt.cm.get_cmap('viridis')
        scatter = ax.scatter(f[:,0],f[:,1],c=arg,s=1,picker=5)
        lines, labels = scatter.legend_elements()
        labels = [str(i) for i in range(10)]
        if legend:
            legend = ax.legend(lines,labels)
            ax.add_artist(legend)            
        
def plot_latent(model_name, split='test',test_path='./test_results', snr=None,p=1, overwrite_tsne=False, mode='pca'):
    if mode == 'pca':
        f = get_pca(model_name, split, test_path, snr, overwrite=overwrite_tsne)
    elif mode == 'tsne':
        f = get_tsne(model_name, split, test_path, snr, overwrite=overwrite_tsne)
    else:
        print('undefined mode: %s'%mode)
    fig=plt.figure(figsize=(15,5))
    
    dataset = CWRUDataset('./data',split,p=p,snr=snr)
    pred = np.load(test_path+'/pred_%s[%s]_%s.npy'%(model_name,snr,split))
    pred=pred[1]
    ax0 = plt.subplot(131)
    ax0.set_title('True Label')
    ax1 = plt.subplot(132)
    ax1.set_title('Prediction')
    ax2=plt.subplot(133)
    ax2.set_title('Difference')
    
    ax0.set_xticks([])
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax0.set_yticks([])
    ax1.set_yticks([])
    ax2.set_yticks([])
    draw_plot(ax0,f,dataset.labels,split=split)
    draw_plot(ax1,f,pred,split=split)
    draw_plot(ax2,f,(pred,dataset),difference=True,split=split)
    plt.show()
    
def draw_magic(model_name, split='test',test_path='./test_results', snr=None,p=1,legend=True, mode='pca'):
    global picked,m,pick
    picked=None
    m=0
    pick=False
    fig=plt.figure(figsize=(5,5))
    
    ax0 = plt.subplot2grid((4,1),(0,0),rowspan=3)#(3,1,1)
    ax1 = plt.subplot(414)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    titles = ['True Label','Prediction','Difference']
    text = ax0.text(0,100,"%s"%titles[m],va='top',ha='left')
    
    dataset = CWRUDataset('./data',split,p=p,snr=snr)
    pred = np.load(test_path+'/pred_%s[%s]_%s.npy'%(model_name,snr,split))
    pred=pred[1]
    if mode == 'pca':
        f = get_pca(model_name, split, test_path, snr)
    elif mode == 'tsne':
        f = get_tsne(model_name, split, test_path, snr)
    else:
        print('undefined mode: %s'%mode)
    
    def onpress(event):
        global m,pick
        if event.key=='right' and m < len(titles)-1:
            m+=1
        elif event.key == 'left' and m > 0:
            m-=1
        else:
            return
        if pick:
            picked.remove()
            pick=False
        ax0.cla()
        ax0.set_xticks([])
        ax0.set_yticks([])
        text = ax0.text(0,100,"%s"%titles[m],va='top',ha='left')
        if m==0:
            arg=dataset.labels
        elif m==1:
            arg=pred
        else:
            arg=(pred,dataset)
        draw_plot(ax0,f,arg,split=split,difference=(m==2),legend=legend)
        fig.canvas.draw()
        
    def onpick(event):
        global picked,pick
        pt = event.ind[0]
        if pick:
            picked.remove()
            pick=False
        picked = ax0.scatter(f[pt,0],f[pt,1],marker='x',s=10,c='r')
        pick=True
        
        ax1.cla()
        ax1.set_xticks([])
        ax1.set_yticks([])
        raw,_,_ = dataset[pt]
        ax1.plot(raw.reshape(-1))
        fig.canvas.draw()
    
    cid = fig.canvas.mpl_connect('pick_event', onpick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onpress)
    draw_plot(ax0, f,dataset.labels,split=split,difference=False,legend=legend)
    plt.show()