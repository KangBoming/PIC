import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

def get_index(data_path,label_name,test_ratio,val_ratio,random_seed):
    with open(data_path,'rb') as f:
        data=pickle.load(f)
    random.seed(random_seed)
    cell_series=data[label_name]
    all_indexes=cell_series.index.to_list()
    all_ess_indexes = [i for i, e in enumerate(data[label_name]) if int(e) == 1]
    all_non_indexes = [i for i, e in enumerate(data[label_name]) if int(e) == 0]
    num_ess = len(all_ess_indexes)
    num_non = len(all_non_indexes)
    weight = num_non / num_ess
    test_indexes = random.sample(all_ess_indexes, int(test_ratio * num_ess)) + random.sample(all_non_indexes, int(test_ratio * num_non))
    random.shuffle(test_indexes)
    train_indexes=list(set(all_indexes)-set(test_indexes))
    val_indexes=random.sample(train_indexes,int(val_ratio * len(train_indexes)))
    real_train_indexes=list(set(train_indexes)-set(val_indexes))
    random.shuffle(real_train_indexes)
    train_cell_series=cell_series.loc[real_train_indexes]
    val_cell_series=cell_series.loc[val_indexes]
    train_dict=[{index:weight} if value == 1 else {index:1.0}
                for index, value in train_cell_series.items()]
    val_dict=[{index:weight} if value == 1 else {index:1.0}
                for index, value in val_cell_series.items()]
    return train_dict,val_dict, test_indexes


class PIC_Dataset(Dataset):
    def __init__(self, indexes,feature_dir,label_name,max_length,feature_length,device) -> None:
        super().__init__()
        self.indexes = indexes
        self.target_dir = feature_dir
        self.max_length=max_length
        self.feature_length=feature_length
        self.cell_name=label_name
        self.device=device
    def load_pt(self, index: int):
        pt_path = f"{self.target_dir}/{index}.pt"
        return torch.load(pt_path,map_location=self.device)
    def __len__(self) -> int:
        return len(self.indexes)
    def __getitem__(self, index: int):
        sample_index = self.indexes[index]
        sample = self.load_pt(sample_index)
        feature = sample['representations'][33].to(self.device)
        label=torch.tensor(eval(sample['label'].split('_')[-1]),dtype=torch.float32).to(self.device)
        seq_length = feature.shape[0]
        if seq_length < self.max_length:
            pad_length = self.max_length - seq_length
            zero_features = torch.zeros((pad_length, self.feature_length)).to(self.device)
            feature = torch.cat([feature, zero_features]).to(self.device)
            start_padding_idx = torch.tensor(seq_length).to(self.device) 
        else:
            feature = feature[:self.max_length]
            start_padding_idx = torch.tensor(-1).to(self.device)
        return feature,label,start_padding_idx
    
    
def make_dataloader(data_path,label_name,test_ratio,val_ratio,
                    random_seed,max_length,feature_length,
                    feature_dir,batch_size,device):
    train_dict ,val_dict, test_indexes=get_index(data_path=data_path,label_name=label_name,
                                                 test_ratio=test_ratio,val_ratio=val_ratio,
                                                 random_seed=random_seed)
    train_indexes=[list(d.keys())[0] for d in train_dict]
    val_indexes=[list(d.keys())[0] for d in val_dict]

    train_dataset_weights=[list(d.values())[0] for d in train_dict]
    val_dataset_weights=[list(d.values())[0] for d in val_dict]
  
    train_dataset=PIC_Dataset(feature_dir=feature_dir,indexes=train_indexes,label_name=label_name,max_length=max_length,feature_length=feature_length,device=device)
    val_dataset=PIC_Dataset(feature_dir=feature_dir,indexes=val_indexes,label_name=label_name,max_length=max_length,feature_length=feature_length,device=device)
    test_dataset=PIC_Dataset(feature_dir=feature_dir,indexes=test_indexes,label_name=label_name,max_length=max_length,feature_length=feature_length,device=device)
   
    train_sampler=WeightedRandomSampler(weights=train_dataset_weights,
                                        num_samples=len(train_dataset),
                                        replacement=True) 
    val_sampler=WeightedRandomSampler(weights=val_dataset_weights,
                                      num_samples=len(val_dataset),
                                      replacement=True)
    train_dataloader=DataLoader(train_dataset,
                                batch_size=batch_size,
                                sampler=train_sampler) 
    val_dataloader=DataLoader(val_dataset,
                                batch_size=batch_size,
                                sampler=val_sampler)
    test_dataloader=DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False) 
    print(f"Length of train dataloader: {len(train_dataloader)} batches of {batch_size}")
    print(f"Length of val dataloader: {len(val_dataloader)} batches of {batch_size}")
    print(f"Length of test dataloader: {len(test_dataloader)} batches of {batch_size}")
    return train_dataloader,val_dataloader,test_dataloader






