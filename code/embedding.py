import os
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pathlib
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='This is a Python script file for extracting protein sequence embedding.')
parser.add_argument('--data_path',type=str,help='The save path of dataset for protein sequence embedding.')
parser.add_argument('--fasta_file',type=str,help='The save path of fasta file for protein sequence.')
parser.add_argument('--model_name',type=str,default='./pretrained_model/esm2_t33_650M_UR50D.pt',help='The protein language model you selected to extract protein sequence embedding.')
parser.add_argument('--label_name',type=str,help='The type of PIC you want to train.')
parser.add_argument('--output_dir',type=str,help='The save directory of protein sequence embedding.')
parser.add_argument('--toks_per_batch',type=int,default=10000, help='If you encounter a CUDA out of memory error, please reduce this value.')
parser.add_argument('--repr_layers',type=list,default=[-1])
parser.add_argument('--device',type=str,default='cuda:0',help='If you use CPU for training, you can input cpu; If you use GPU for training, you can enter cuda & GPU number, for example: cuda:0.')
parser.add_argument('--include',type=list,default=['mean','per_tok'])
parser.add_argument('--truncation_seq_length',type=int,default=1024)
args= parser.parse_args()


def generate_fasta(data_path,save_path, label_name):

    dataset=pd.read_pickle(data_path)
    with open(save_path,'w') as fasta:
            for index, row in dataset.iterrows():
                sequence=row['sequence'].replace('*','')
                id=str(row['index'])+'_'+row['ID']+'_'+str(row[f'{label_name}'])
                fasta.write(f'>{id}\n{sequence}\n')
    return 'FASTA file has been genrated'


def extract_embedding(model_name, fasta_file,
                      output_dir, toks_per_batch, 
                      repr_layers, device, 
                      include, truncation_seq_length):
    # Load pretrained protein language model

    ''' 
    tips:
    If you wish to download a pretrained model here, you can set 'model_name' to the name of ESM model, e.g. esm2_t33_650M_UR50D
    If you have already downloaded the pretrained model locally, set 'model_name' to the file path of ESM model, e.g. ./pretrained_model/esm2_t33_650M_UR50D.pt
    '''

    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
    if torch.cuda.is_available():
        model = model.to(device=device)
        print("Transferred model to GPUs")
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")
    return_contacts = "contacts" in include
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]


    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            toks = toks.to(device=device if torch.cuda.is_available() else "cpu", non_blocking=True)
            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
            logits = out["logits"].to(device=device if torch.cuda.is_available() else "cpu")
            representations = {
                layer: t.to(device=device) for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].clone()
            for i, label in enumerate(labels):
                output_file = pathlib.Path(output_dir) / f"{label.split('_')[0]}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(truncation_seq_length, len(strs[i]))
                if "per_tok" in include:
                    result["representations"] = {
                        layer: t[i, 1:truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in include:
                    result["mean_representations"] = {
                        layer: t[i, 1:truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, :truncate_len, :truncate_len].clone()
                torch.save(result, output_file)
    return 'Embedding has been extracted'


def main():
    generate_fasta(data_path=args.data_path,
                   save_path=args.fasta_file,
                   label_name= args.label_name)
    extract_embedding(
        model_name=args.model_name,
        fasta_file=args.fasta_file,
        output_dir=args.output_dir,
        toks_per_batch=args.toks_per_batch,
        repr_layers=args.repr_layers,
        device=args.device,
        include=args.include,
        truncation_seq_length=args.truncation_seq_length
    )


if __name__ == '__main__':
        main()




