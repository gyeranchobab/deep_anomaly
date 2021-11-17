import matplotlib.pyplot as plt
import os
import pickle
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_logs(model_names):
    AX_NAME = ['Train Loss','Test Loss']#
    
    fig=plt.figure(figsize=(10,5*len(AX_NAME)//2))
    
    log_path = './train_logs'
    for m in model_names:
        if not os.path.isfile(log_path+'/%s.bin'%m):
            print('No %s'%m)
            continue
        results = pickle.load(open(log_path+'/%s.bin'%m,'rb'))
        ax = plt.subplot(121)
        ax.set_title(AX_NAME[0])
        ax.plot(results['train_loss'], label=m)
        ax.legend()
        
        ax = plt.subplot(122)
        ax.set_title(AX_NAME[1])
        ax.plot(results['eval_loss'], label=m)
        ax.legend()
        '''
        for i,r in enumerate(results):
            print(r)
            #if i >= len(AX_NAME):
            #    break
            ax = plt.subplot(len(AX_NAME)//2,2,i+1)
            ax.set_title(AX_NAME[i])
            ax.plot(r, label=m)
            if i<1:
                ax.legend()'''
    plt.show()
    
def plot_latent(model, epoch, test_loader, method='pca', split='test'):
    if not os.path.isfile('./saved_models/%s_e%d.pth'%(model.model_name, epoch)):
        print("No saved model for %s, epoch %d"%(model.model_name, epoch))
        return
    model.load_state_dict(torch.load('./saved_models/%s_e%d.pth'%(model.model_name, epoch)))
    
    zs = []
    preds = []
    labels = []
    masks = []
    model.eval()
    for batch, label, mask in test_loader:
        batch=batch.cuda().to(dtype=torch.float)
        label=label.cuda()#.to(dtype=torch.long)
        if label.dtype == torch.int32:
            label = label.to(dtype=torch.long)
        elif label.dtype == torch.double:
            label = label.to(dtype=torch.float)
        mask=mask.cuda().to(dtype=torch.bool)
        batch = model.preprocess(batch, label, mask)
        with torch.no_grad():
            pred, z = model.get_z(batch, label, mask)
        pred = torch.argmax(pred, dim=1)
        preds.append(pred.cpu())
        zs.append(z.cpu())
        labels.append(label.cpu())
        masks.append(mask.cpu())
    zs = torch.cat(zs, 0)
    preds = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)
    masks = torch.cat(masks, 0)
    #print(zs.shape)
    #print(preds.shape)
    reducer = None
    if method == 'pca':
        reducer = PCA(2)
    elif method == 'tsne':
        reducer = TSNE(2)
    else:
        print("method should be either ['pca' or 'tsne']")
        return
    print('Start Reducing Z Dimension for Visualization')
    #print(zs.min(), zs.max(), zs.mean())
    reduced_z = reducer.fit_transform(zs)
    
    fig = plt.figure(figsize=(15,5))
    
    ax0 = plt.subplot(131)
    ax0.set_title('True Label')
    ax0.axis('off')
    ax1 = plt.subplot(132)
    ax1.set_title('Prediction')
    ax1.axis('off')
    ax2 = plt.subplot(133)
    ax2.set_title('Difference')
    ax2.axis('off')
    
    
    draw_plot(ax0, reduced_z, labels, split=split)
    draw_plot(ax1, reduced_z, preds, split=split)
    draw_plot(ax2, reduced_z, (preds,labels,masks), difference=True, split=split)
    #plt.scatter(reduced_z[:,0], reduced_z[:,1])
    #print(reduced_z.shape)
    
    plt.show()

def draw_plot(ax, f, arg, difference=False, split='test',legend=True):
    if difference:
        pred, labels, masks = arg
        #print(labels.shape, pred.shape, labels==pred)
        scatter = ax.scatter(f[:,0],f[:,1],c=labels==pred,s=1,alpha=0.5,picker=5,cmap='Set1')
        lines, labels = scatter.legend_elements()
        labels = ['Misclassified','Correct']
        if split == 'train':
            scatter = ax.scatter(f[:,0][masks],f[:,1][masks],c='blue',s=5,label='Labeled')
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
            
def make_confmat(model, epoch, test_loader):
    if not os.path.isfile('./saved_models/%s_e%d.pth'%(model.model_name, epoch)):
        print("No saved model for %s, epoch %d"%(model.model_name, epoch))
        return
    model.load_state_dict(torch.load('./saved_models/%s_e%d.pth'%(model.model_name, epoch)))
    
    preds = []
    labels = []
    model.eval()
    for batch, label, mask in test_loader:
        batch=batch.cuda().to(dtype=torch.float)
        label=label.cuda()#.to(dtype=torch.long)
        if label.dtype == torch.int32:
            label = label.to(dtype=torch.long)
        elif label.dtype == torch.double:
            label = label.to(dtype=torch.float)
        mask=mask.cuda().to(dtype=torch.bool)
        batch = model.preprocess(batch, label, mask)
        with torch.no_grad():
            pred = model(batch, label, mask)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred = torch.argmax(pred, dim=1)
        preds.append(pred.cpu())
        labels.append(label.cpu())
    preds = torch.cat(preds, 0)
    labels = torch.cat(labels, 0)
    
    matrix = [[len(labels[(labels==i)&(preds==j)]) for j in range(5)] for i in range(5)]
    fig,ax = plt.subplots()
    im = ax.imshow(matrix, cmap='viridis')
    ax.set_xlabel('Pred')
    ax.set_ylabel('True')
    for i in range(5):
        for j in range(5):
            text = ax.text(j,i,matrix[i][j],ha='center',va='center',color='w')
    fig.tight_layout()
    plt.show()