import random
import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score ,precision_score,recall_score,roc_auc_score,average_precision_score,f1_score
from module.load_dataset import get_index,make_dataloader,PIC_Dataset
from module.PIC import PIC
from module.loss_func import FocalLoss_L2
from module.earlystopping import EarlyStopping



parser = argparse.ArgumentParser(description='This is a Python script file for PIC model training.')
parser.add_argument('--device',type=str,default='cuda:0',help='If you use CPU for training, you can input cpu; If you use GPU for training, you can enter cuda & GPU number, for example: cuda:0.')
parser.add_argument('--data_path',type=str,help='The save path of dataset for model training.')
parser.add_argument('--label_name',type=str,help='The type of PIC you want to train.')
parser.add_argument('--test_ratio',type=float,default=0.1)
parser.add_argument('--val_ratio',type=float,default=0.1)
parser.add_argument('--feature_dir',type=str,help='The save directory of protein sequence embedding.')
parser.add_argument('--batch_size',type=int,default=256)
parser.add_argument('--save_path',type=str,help='The save directory for model training results. ')
parser.add_argument('--linear_drop',type=float,default=0.1)
parser.add_argument('--attn_drop',type=float,default=0.3)
parser.add_argument('--max_length',type=float,default=1000)
parser.add_argument('--feature_length',type=float,default=1280)
parser.add_argument('--learning_rate',type=float,default=1e-5)
parser.add_argument('--input_size',type=float,default=1280)
parser.add_argument('--hidden_size',type=float,default=320)
parser.add_argument('--output_size',type=float,default=1)
parser.add_argument('--num_epochs',type=float,default=15)
parser.add_argument('--random_seed',type=float,default=42)
args= parser.parse_args()


def model_train(dataloader,model,loss_fn,optimizer,device):
    model.train()
    train_loss , train_acc=0 , 0 
    for batch , (X,y,index) in enumerate(dataloader):
        X,y,index=X.to(device) ,y.to(device),index.to(device)
        y=y.unsqueeze(1)
        train_logits=model(X,index)
        train_pred=torch.round(torch.sigmoid(train_logits))
        loss=loss_fn(train_logits,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss=loss.cpu().detach().numpy()
        train_loss += loss
        train_acc=accuracy_score(y_true=y.squeeze().cpu().detach().numpy(),y_pred=train_pred.squeeze().cpu().detach().numpy())   
    average_train_loss= train_loss / len(dataloader)
    average_train_acc= train_acc / len(dataloader)
    return average_train_loss , average_train_acc


def model_val(dataloader,model,loss_fn,device,mode):
    val_loss=0
    val_preds=[]
    val_pred_scores=[] 
    val_targets=[] 
    model.eval()
    with torch.inference_mode():
        for X,y,index in dataloader:
            y=y.unsqueeze(1)
            X,y,index=X.to(device),y.to(device),index.to(device)
            val_logits=model(X,index)
            y_score=torch.sigmoid(val_logits) 
            val_pred=torch.round(torch.sigmoid(val_logits))
            loss=loss_fn(val_logits,y)
            val_preds.append(val_pred.cpu().numpy())
            val_pred_scores.append(y_score.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_loss += loss.item()
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_pred_scores=np.concatenate(val_pred_scores)
    average_val_loss= val_loss / len(dataloader)
    average_val_acc = accuracy_score(y_true=val_targets,y_pred=val_preds)
    average_val_recall = recall_score(y_true=val_targets,y_pred=val_preds)
    average_val_roc_auc = roc_auc_score(y_true=val_targets,y_score=val_pred_scores)
    average_val_prec = precision_score(y_true=val_targets,y_pred=val_preds)
    average_val_f1 = f1_score(y_true=val_targets,y_pred=val_preds)
    average_val_pr_auc=average_precision_score(y_true=val_targets,y_score=val_pred_scores)

    print(f"{mode} loss: {average_val_loss:>6f} \n")
    print(f"{mode} acc: {average_val_acc:>6f}",
          f"{mode} recall: {average_val_recall:>6f}",
          f"{mode} roc_auc: {average_val_roc_auc:>6f}",
          f"{mode} precision: { average_val_prec:>6f}",
          f"{mode} pr_auc: { average_val_pr_auc:>6f}")
    return average_val_loss , average_val_acc,average_val_recall,average_val_roc_auc,average_val_prec,average_val_f1,average_val_pr_auc


def train_val_loop(model,epochs,train_data_loader,val_data_loader,
                   loss_fn,optimizer,model_name,
                   mode,scheduler,earlystopping,device):
    result_dict={}
    epoch_count=[]
    train_loss_values=[]
    train_acc_values=[]
    val_loss_values=[]
    val_acc_values=[]
    val_recall_values=[]
    val_auc_values=[]
    val_prec_values=[]
    val_f1_values=[]
    val_pr_auc_values=[]
    if mode == 'val':
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            epoch_count.append(epoch+1)
            train_loss,train_acc=model_train(model=model,dataloader=train_data_loader,loss_fn=loss_fn,optimizer=optimizer,device=device)
            train_loss_values.append(train_loss)
            train_acc_values.append(train_acc)
            val_loss,val_acc,val_recall,val_roc_auc,val_prec,val_f1,val_pr_auc=model_val(model=model,dataloader=val_data_loader,loss_fn=loss_fn,device=device,mode='val')
            val_loss_values.append(val_loss)
            val_acc_values.append(val_acc)
            val_recall_values.append(val_recall)
            val_auc_values.append(val_roc_auc)
            val_prec_values.append(val_prec)
            val_f1_values.append(val_f1)
            val_pr_auc_values.append(val_pr_auc)
            scheduler.step(val_loss)
            if earlystopping(val_loss):
                print(f"Early stop in epoch {epoch+1}\n")
                break
        print('Done!')
        result_dict[model_name + '_' + 'epoch'] = epoch_count
        result_dict[model_name + '_' + 'train_loss'] = train_loss_values
        result_dict[model_name + '_' + 'train_acc'] = train_acc_values
        result_dict[model_name + '_' + 'val_loss'] = val_loss_values
        result_dict[model_name + '_' + 'val_acc'] = val_acc_values
        result_dict[model_name + '_' + 'val_recall'] = val_recall_values
        result_dict[model_name + '_' + 'val_auc'] = val_auc_values
        result_dict[model_name + '_' + 'val_prec'] = val_prec_values
        result_dict[model_name + '_' + 'val_f1'] = val_f1_values
        result_dict[model_name + '_' + 'val_pr_auc'] = val_pr_auc_values
        result_df=pd.DataFrame(result_dict)
    elif mode == 'test':
        test_loss,test_acc,test_recall,test_roc_auc,test_prec,test_f1,test_pr_auc=model_val(model=model,dataloader=val_data_loader,loss_fn=loss_fn,device=device,mode='test')
        result_dict[model_name + '_' + 'test_loss'] = [test_loss]
        result_dict[model_name + '_' + 'test_acc'] = [test_acc]
        result_dict[model_name + '_' + 'test_recall'] = [test_recall]
        result_dict[model_name + '_' + 'test_auc'] = [test_roc_auc]
        result_dict[model_name + '_' + 'test_prec'] = [test_prec]
        result_dict[model_name + '_' + 'test_f1'] = [test_f1]
        result_dict[model_name + '_' + 'test_pr_auc'] = [test_pr_auc]
        result_df=pd.DataFrame(result_dict)
    else:
        print(f'Please input val or test')
    return result_df


def model_train_main(data_path,label_name,
                     test_ratio,val_ratio,
                     feature_dir,batch_size,save_path,
                     linear_drop,attn_drop,
                     max_length,feature_length,
                     learning_rate,input_size,hidden_size,
                     output_size,device,
                     num_epochs,random_seed):
    train_dataloader,val_dataloader,test_dataloader=make_dataloader(data_path=data_path,
                                                            test_ratio=test_ratio,
                                                            val_ratio=val_ratio,
                                                            max_length=max_length,
                                                            feature_length=feature_length,
                                                            random_seed=random_seed,
                                                            feature_dir=feature_dir,
                                                            label_name=label_name,
                                                            batch_size=batch_size,
                                                            device=device)

    model = PIC(input_shape=input_size,hidden_units=hidden_size,device=device,
                linear_drop=linear_drop,attn_drop=attn_drop,output_shape=output_size)
    model=model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,weight_decay=1e-4)
    loss_fn =FocalLoss_L2(gamma=0, pos_weight=1.0, logits=True, reduction='mean',weight_decay=1e-4)
    earlystopping = EarlyStopping(mode='min', patience=5) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, threshold=0.0001, verbose=True)
    val_result_df=train_val_loop(mode='val',
    model=model,
    model_name=label_name,
    train_data_loader=train_dataloader,val_data_loader=val_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    earlystopping=earlystopping,
    epochs=num_epochs,device=device)
    test_result_df=train_val_loop(mode='test',
    model=model,
    model_name=label_name,
    train_data_loader=train_dataloader,
    val_data_loader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    earlystopping=earlystopping,
    epochs=num_epochs,device=device)
    file_name=f'PIC_{label_name}'
    model_folder = os.path.join(save_path, f"{file_name}")
    os.makedirs(model_folder, exist_ok=True)
    model_path = os.path.join(model_folder, f"{file_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    val_result_path = os.path.join(model_folder, f"{file_name}_val_result.csv")
    val_result_df.to_csv(val_result_path, index=False)
    test_result_path = os.path.join(model_folder, f"{file_name}_test_result.csv")
    test_result_df.to_csv(test_result_path, index=False)
    print(f'{file_name}_model has done')


def main():
    model_train_main(
        data_path=args.data_path,
        label_name=args.label_name,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        feature_dir=args.feature_dir,
        batch_size=args.batch_size,
        save_path=args.save_path,
        linear_drop=args.linear_drop,
        attn_drop=args.attn_drop,
        max_length=args.max_length,
        feature_length=args.feature_length,
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        device=args.device,
        num_epochs=args.num_epochs,
        random_seed=args.random_seed
    )


if __name__ == '__main__':
    main()


