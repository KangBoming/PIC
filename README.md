# PIC:Protein Importance Calculator
PIC is a sequence-based model for multi-level essential protein prediction. PIC utilizes a pre-trained protein language model to extract sequence features. Then, we add a multi-head self-attention layer to capture the relative importance of amino acids at different positions. Finally, a multi-layer perceptron is used to predict essential proteins.

The PIC web server for comprehensive prediction and analysis of the essentiality of human proteins is now available at http://www.cuilab.cn/
## Requirements
* python=3.10.13
* pytorch=1.12.1
* torchaudio=0.12.1
* torchvision=0.13.1
* cudatoolkit=11.3.1
* scikit-learn=1.3.2
* pandas=2.1.1
* numpy=1.26.0
* fair-esm=2.0.0
## Usage
A demo for training PIC-cell model using linux-64 platform

**Step1: clone the repo**
```
git clone https://github.com/KangBoming/PIC.git
cd PIC
```
**Step2: create and activate the environment**
```
cd PIC
conda create --name PIC --file requirments.txt
conda activate PIC
```
**Step3: extract the sequence embedding from raw protein sequences** 
```
cd PIC
python ./code/embedding.py --data_path './data/PIC_cell_dataset.pkl' --fasta_file './result/protein_sequence.fasta' --model 'esm2_t33_650M_UR50D' --ouput_dir './result/seq_embedding' --device 'cuda:0' --truncation_seq_length 1024

```


## License

## Contact
