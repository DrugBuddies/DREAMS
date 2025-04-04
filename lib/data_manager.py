import torch
from datasets import load_from_disk
from geneformer import TOKEN_DICTIONARY_FILE
from torch.utils.data import Dataset
from geneformer import TranscriptomeTokenizer
import pickle as pkl
import geneformer.perturber_utils as pu
import json

from transformers import AutoTokenizer


def tokenize_h5ad(in_dir, out_dir, file_format='h5ad', key_name = 'cell_line'):
    """
    convert single cell h5ad dataset  to dataset compatible with Dreams
    :param in_dir: path to h5ad dataset
    :param out_dir: path to save dataset
    :param file_format: h5ad or loom
    :param key_name:  name of the key to be added
    :return:
    """
    tok = TranscriptomeTokenizer({key_name:key_name}, special_token=True)
    tok.tokenize_data(in_dir, out_dir, "temp", file_format=file_format)

def sample_pairs(drug_data, cell_data, n_pairs_per_sample):
    # TODO this function should build cell-drug pairs
    return []

def perturb(input_data, perturbation, token_dict):
    """
    perturb a tokenized cell
    :param input_data: tokenized cell data
    :param perturbation: dict that maps gene to perturbation type
    :param token_dict: dictionary of the geneformer
    :return:
    """
    perturbed = input_data.copy()
    for k in perturbation.keys():
        if k not in token_dict.keys():
            continue
        gene_token = token_dict[k]
        to_append = torch.tensor([gene_token],dtype=torch.int)
        perturbed['input_ids'] = perturbed['input_ids'][ perturbed['input_ids'] != gene_token]
        if perturbation[k] == 'agonist':
            perturbed['input_ids'] = torch.concat(( perturbed['input_ids'][:1], # cls token
                                                       to_append ,   # gene to overexpress
                                                           perturbed['input_ids'][1:])) # other genes
    perturbed['length'] = torch.tensor(len(perturbed['input_ids']))
    return perturbed


class DREAMSDataset(Dataset):

    def __init__(self,
                 drugs_data_path,
                 model_input_size,
                 temp_dir='data/temp',
                 file_format='h5ad',
                 n_pairs_per_sample=1):

        with open(TOKEN_DICTIONARY_FILE, "rb") as f:
            self.gene_token_dict = pkl.load(f)

        self.model_input_size = model_input_size
        self.transcriptome = load_from_disk(temp_dir)
        self.transcriptome.set_format('torch')
        self.pad_token_id = self.gene_token_dict.get("<pad>")

        with open(drugs_data_path, "r") as f:
            self.drugs = json.load(f) # dictionary of drugs

        self.drug_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

        self.indices = sample_pairs(self.transcriptome, self.drugs, n_pairs_per_sample)


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        cell_idx, drug_id = self.indices[index]
        cell = self.transcriptome[cell_idx]
        smile = self.drugs[drug_id]['SMILES']
        perturbation = perturb(cell, self.drugs[drug_id]['targets'], self.gene_token_dict )
        return cell, perturbation, smile

    def tokenize_cells(self, cells):
        input_batch = cells['input_ids']
        max_len = max(cells["length"])
        input_batch =  pu.pad_tensor_list(
            input_batch, int(max_len), self.pad_token_id, self.model_input_size
        )

        attn = pu.gen_attention_mask(cells, max_len)
        return input_batch, attn

    def tokenize_drugs(self, smiles):
        return self.drug_tokenizer(smiles, padding=True, return_tensors="pt")

