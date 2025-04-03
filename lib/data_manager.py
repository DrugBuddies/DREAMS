import torch
from datasets import load_from_disk
from geneformer import TOKEN_DICTIONARY_FILE
from torch.utils.data import DataLoader
from geneformer import TranscriptomeTokenizer
import pickle as pkl
import geneformer.perturber_utils as pu
import json

from transformers import AutoTokenizer



def perturb(input_data, perturbation, token_dict):
    perturbed = input_data.copy()
    for i in range(len(input_data['input_ids'])):
        for k in perturbation.keys():
            gene_token = token_dict[k]
            to_append = torch.tensor([gene_token],dtype=torch.int)
            perturbed['input_ids'][i] = perturbed['input_ids'][i][ perturbed['input_ids'][i] != gene_token]
            if perturbation[k] == 'overexpress':
                perturbed['input_ids'][i] = torch.concat(( perturbed['input_ids'][i][:1], # cls token
                                                           to_append ,   # gene to overexpress
                                                           perturbed['input_ids'][i][1:])) # other genes
    return perturbed


class DREAMSDataLoader(DataLoader):

    def __init__(self, transcriptome_path,
                 drugs_data_path,
                 model_input_size,
                 temp_dir='data/temp',
                 file_format='h5ad'):


        # load and tokenize transcriptomics data
        tok = TranscriptomeTokenizer({'tissue':'tissue'}, special_token=True)
        tok.tokenize_data(transcriptome_path, temp_dir, "temp", file_format=file_format)


        with open(TOKEN_DICTIONARY_FILE, "rb") as f:
            self.gene_token_dict = pkl.load(f)

        self.model_input_size = model_input_size
        self.dataset = load_from_disk(temp_dir)
        self.dataset.set_format('torch')
        self.pad_token_id = self.gene_token_dict.get("<pad>")

        with open(drugs_data_path, "r") as f:
            self.drugs = json.load(f) # dictionary of drugs

        self.drug_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

        self.indices = []  #  ( i, 'id')


        # TODO: sposta qua codice per preprocessare
        # TODO: compute pairs cell-drug based on tissue
        # TODO:

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        cell_idx, drug_id = self.indices[index]
        cell = self.dataset[cell_idx]
        smile = self.drugs[drug_id]['smile']
        perturbation = perturb(cell, self.drugs[drug_id]['perturbations'], self.gene_token_dict )
        return cell, perturbation, smile

    def tokenize_cells(self, cells):
        input_batch = cells['input_ids']
        max_len = int(max(cells["length"]))
        input_batch =  pu.pad_tensor_list(
            input_batch, max_len, self.pad_token_id, self.model_input_size
        )
        attn = pu.gen_attention_mask(input_batch, max_len)
        return input_batch, attn

    def tokenize_drugs(self, smiles):
        smile_inputs = self.drug_tokenizer(smiles, padding=True, return_tensors="pt")
        return smile_inputs['input_ids'], smile_inputs['attention_mask']


    def sample_pairs(self):
        # TODO this function should build cell-drug pairs
        pass


def sample_example(idx, dataset):
    c,p,d =  dataset[idx]
    c_tok = dataset.tokenize_cells(c)
    p_tok = dataset.tokenize_cells(p)
    d_tok = dataset.tokenize_drugs(d)