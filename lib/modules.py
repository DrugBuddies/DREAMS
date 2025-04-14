import geneformer.perturber_utils as pu
import torch
from peft import LoraConfig, get_peft_model
from torch.functional import F
from torch import nn
from transformers import (
    BertForMaskedLM,
    AutoModel, )

class CLIPLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature

    def forward(self, drug_emb, perturb_emb):
        # Calculating the Loss
        logits = (drug_emb @ perturb_emb.T) / self.temperature
        drug_sim = drug_emb @ drug_emb.T
        perturb_sim = perturb_emb @ perturb_emb.T
        targets = F.softmax(
            (drug_sim + perturb_sim) / 2 * self.temperature, dim=-1
        )
        drug_loss = F.cross_entropy(logits, targets, reduction='none')
        perturb_loss = F.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (drug_loss + perturb_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


class DREAMSEncoder(nn.Module):

    def __init__(self,
                 proj_shape = 256,
                 molformer_emb_shape = 768,
                 geneformer_emb_shape = 256,
                 device='cuda'):
        super(DREAMSEncoder, self).__init__()

        self.geneformer = TrainablePerturber().to(device)
        self.molformer = TrainableMolformer().to(device)
        self.proj_shape = proj_shape
        self.molformer_hidden_size = (molformer_emb_shape + proj_shape) // 2
        self.geneformer_hidden_size = (geneformer_emb_shape + proj_shape) // 2

        self.geneformer_proj_head = nn.Sequential(
            nn.Linear(geneformer_emb_shape, self.geneformer_hidden_size),
            nn.Linear(self.geneformer_hidden_size, proj_shape)
        ).to(device)

        self.molformer_proj_head = nn.Sequential(
            nn.Linear(molformer_emb_shape, self.molformer_hidden_size),
            nn.Linear(self.molformer_hidden_size, proj_shape)
        ).to(device)

    def forward(self, geneformer_input, molformer_input):
        cell_emb =  self.geneformer(**geneformer_input)
        drug_emb = self.molformer(**molformer_input)

        cell_emb = self.geneformer_proj_head(cell_emb)
        drug_emb = self.molformer_proj_head(drug_emb)

        return cell_emb,drug_emb


    def embed_drug(self, molformer_input):
        with torch.no_grad():
            out =  self.molformer(**molformer_input)
            return self.molformer_proj_head(out)

    def embed_perturbation(self, geneformer_input):
        with torch.no_grad():
            out = self.geneformer(**geneformer_input)
            return self.geneformer_proj_head(out)






class TrainableMolformer(nn.Module):
    def __init__(self):
        super(TrainableMolformer, self).__init__()

        model_args = {
            "pretrained_model_name_or_path": "ibm/MoLFormer-XL-both-10pct",
            "output_hidden_states": True,
            "output_attentions": False,
            "deterministic_eval": True,
            "trust_remote_code":True,
        }


        self.model = AutoModel.from_pretrained(**model_args)

        peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=["query", "key", "value"]
        )
        self.model.enable_input_require_grads()
        self.model = get_peft_model(self.model, peft_config)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids,attention_mask=attention_mask)['last_hidden_state'].mean(axis=-2)




class TrainablePerturber(nn.Module):

    def __init__(self ):

        super().__init__()
        model_args = {
            "pretrained_model_name_or_path": 'Geneformer/gf-6L-30M-i2048',
            "output_hidden_states": True,
            "output_attentions": False,

        }

        self.model = BertForMaskedLM.from_pretrained(**model_args)



        peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=["query", "key", "value"]
        )
        self.model.enable_input_require_grads()
        self.model = get_peft_model(self.model, peft_config)

        self.max_len = pu.get_model_input_size(self.model)
        self.emb_layer = -1
        self.layer_to_quant = pu.quant_layers(self.model) + self.emb_layer


    def forward(self, x, x_attn, y, y_attn):
        # embed start state
        outputs = self.model(
            input_ids=x,
            attention_mask=x_attn,
        ).hidden_states[self.layer_to_quant]

        x_cls_embs = outputs.mean(axis=0)

        # embed shift state
        outputs = self.model(
            input_ids=y,
            attention_mask=y_attn,
        ).hidden_states[self.layer_to_quant]

        y_cls_embs = outputs.mean(axis=0)

        return y_cls_embs - x_cls_embs


