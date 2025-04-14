import torch
from datasets import load_from_disk
from geneformer import TOKEN_DICTIONARY_FILE
from torch.utils.data import Dataset
from geneformer import TranscriptomeTokenizer
import geneformer.perturber_utils as pu
import pickle as pkl
import random
random.seed(42)
from tqdm import tqdm
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
    # instanciate tokenizer
    tok = TranscriptomeTokenizer({key_name:key_name}, special_token=True)
    # tokenize and save dataset
    tok.tokenize_data(in_dir, out_dir, "tokenized", file_format=file_format)

def sample_pairs(drug_data, cell_data, n_drugs, n_cells_per_drug):
    """
    sample data for an epoch
    :param drug_data: drug dataset
    :param cell_data:  cell dataset
    :param n_drugs: number of drugs-cellline combos to use
    :param n_cells_per_drug:  number of cells for drugs-celline combos
    :return:
    """
    data_list = []
    for _ in tqdm(range(n_drugs),desc='sampling'): # iterate for n times
        cl = random.choice(list(drug_data.keys()))  # choose random cell line
        drug_idx = random.choice(range(len(drug_data[cl])))  # choose random drug
        cells_idxs =  random.sample([   i for i in range(cell_data.shape[0])  # choose ranodm cells
                                        if cell_data[i]['cell_line'].lower().replace('-','') ==
                                        cl.lower().replace('-',' ') ],
                                    n_cells_per_drug)
        data_list += [ (cl, drug_idx, idx)  for idx in cells_idxs ]  # add to data list

    return data_list  # return

def perturb(input_data, genes_up, genes_down, token_dict):
    """
    perturb a tokenized cell
    :param input_data: tokenicedd cell to perturb
    :param genes_down:  list of ensemblids to downregulate
    :param genes_up: list of ensemblids to upregulate
    :param token_dict: dictionary of the geneformer tokenizer
    :return:
    """
    perturbed = input_data.detach().clone()

    for gene in genes_down:  # for each gene to downregulate
        if gene not in token_dict.keys():
            continue  # ignore gene not in tokenizer
        gene_token = token_dict[gene]  # get token id
        perturbed = perturbed[  perturbed != gene_token  ] # remove from tensor

    for gene in genes_up:  # for each gene to upregulate
        if gene not in token_dict.keys():
            continue # ignore gene not in tokenizer
        gene_token = token_dict[gene]  # get token id
        perturbed = perturbed[  perturbed != gene_token  ] # remove from tensor
        to_append = torch.tensor([gene_token],dtype=torch.int)  # build a new tensor containing only the up gene
        perturbed = torch.concat((  perturbed[:1], to_append, perturbed[1:] ))  # put up gene after bos token
    return perturbed



def gen_attention_mask(lens):
    """
    coompute attention masks
    :param lens: list of lengths of samples
    :return:  attention masks
    """
    max_len = max(lens)
    attention_mask = [
        [1] * original_len + [0] * (max_len - original_len)
        if original_len <= max_len
        else [1] * max_len
        for original_len in lens
    ]
    return torch.tensor(attention_mask)


class DREAMSCollator:

    def __init__(self, dataset, device='cuda'):
        """
        Callable class that collate a set of sample of DREAMSDataset
        :param dataset: dataset to collate
        :param device: device to move batches to
        """
        self.pad_token_id = dataset.pad_token_id
        self.model_input_size = dataset.model_input_size
        self.drug_tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct",
                                                            trust_remote_code=True)
        self.device = device

    def __call__(self, batch):
        """
        collate function
        :param batch: batch returned from dataloader of DREAMSDataset
        :return: data in input format for dreams2
        """
        cells = [item[0] for item in batch]
        perturbations = [item[1] for item in batch]
        smiles = [item[2] for item in batch]

        cell_lens = [len(c) for c in cells]
        pert_lens = [len(c) for c in perturbations]


        cell_batch =  pu.pad_tensor_list(  # pad cells
            cells, max(cell_lens), self.pad_token_id, self.model_input_size
        )

        cell_attn_mask = gen_attention_mask(cell_lens)  # get cells attention mask


        pert_batch =  pu.pad_tensor_list(  # pad perturbations
            perturbations, max(pert_lens), self.pad_token_id, self.model_input_size
        )
        pert_attn_mask = gen_attention_mask(pert_lens)  # get perturbation attention mask

        # tokenize smiles
        tokenized_smiles = self.drug_tokenizer(smiles, padding=True, return_tensors="pt")


        return {  # ready to be forwarded to the model
            'x': cell_batch.T.to(self.device),
            'x_attn': cell_attn_mask.T.to(self.device),
            'y': pert_batch.T.to(self.device),
            'y_attn': pert_attn_mask.T.to(self.device),
        }, tokenized_smiles.to(self.device)




class DREAMSDataset(Dataset):
    """
    This class handle dataset of cell-perturbation-drugs triples
    """

    def __init__(self,
                 drugs_data_path,
                 model_input_size,
                 tokenized_sc_path,
                 n_drugs = 200,
                 n_cells_per_drug=4):
        """
        initialization of Dataset
        :param drugs_data_path:  path of drugs pickled dataset
        :param model_input_size:  input size of model
        :param tokenized_sc_path: tokenized transcriptomic dataset
        :param n_drugs: number of drugs-cellline combos to use for each epoch
        :param n_cells_per_drug: number of cells for drugs-celline combos
        """

        # read gene tokenizer dictionary
        with open(TOKEN_DICTIONARY_FILE, "rb") as f:
            self.gene_token_dict = pkl.load(f)

        # set dataset attributes
        self.model_input_size = model_input_size
        self.transcriptome = load_from_disk(tokenized_sc_path)
        self.transcriptome.set_format('torch')
        self.pad_token_id = self.gene_token_dict.get("<pad>")

        # read drugs data
        with open(drugs_data_path, "rb") as f:
            self.drugs = pkl.load(f) # dictionary of drugs

        # sample dataset
        self.indices = sample_pairs(self.drugs, self.transcriptome, n_drugs, n_cells_per_drug)



    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        return a sample of the dataset
        :param index: index of the sample
        :return:  the cell and perturbation token ids, and the drug smile
        """

        cell_line, drug_idx, cell_idx = self.indices[index]  # get sample
        cell = self.transcriptome[cell_idx]['input_ids']  # get cell to perturb
        smile = self.drugs[cell_line][drug_idx]['smiles']  # get smile of the drug
        up = self.drugs[cell_line][drug_idx]['gene_up']  # get genes to upregulate
        down = self.drugs[cell_line][drug_idx]['gene_down']   # get genes to downregulate
        perturbation = perturb(cell, up, down, self.gene_token_dict )  # perturb cell
        return cell, perturbation, smile  # return sample


