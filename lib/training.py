import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(model, loss_fn, train_data, test_data,
          batch_size=4, lr=.001,
          n_epochs=200, device='cuda'):

    model.to(device)
    writer = SummaryWriter()

    # INITIALIZE OPTIMIZER AND DATA CLASSES

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    print_network_to_tensorboard(model, writer, data_loader, train_data)


    optimizer = Adam(model.parameters(), lr=lr)
    min_eval_loss = float('inf')
    best_epoch = 0
    for epoch in range(n_epochs):  # iterate over epochs
        model.train()  # set train mode
        total_loss = 0
        for batch_idx, (cell, pert, drug) in tqdm(enumerate(data_loader),  # iterate over train batches
                                                  total=len(data_loader),
                                                  desc=f'epoch {epoch}: Training'):


            optimizer.zero_grad()  # zero gradients
            c_ids, c_attn = train_data.tokenize_cells(cell)  # preprocess cells
            p_ids, p_attn = train_data.tokenize_cells(pert)  # prerpcoess perturbations
            d_tok = train_data.tokenize_drugs(drug)  # prerpocess drugs

            emb_pert, emb_drug = model({  # forward model
                'x': c_ids.to('cuda'),
                'x_attn': c_attn.to('cuda'),
                'y': p_ids.to('cuda'),
                'y_attn': p_attn.to('cuda')
            }, d_tok.to('cuda'))

            curr_loss = loss_fn(emb_drug, emb_pert )  # compute loss
            curr_loss.backward()  # backpropagation
            optimizer.step()  # update weights
            total_loss += curr_loss.item()

        # log train loss
        avg_loss = total_loss / len(data_loader)  # average loss
        writer.add_scalar('train_loss', avg_loss, epoch) # print loss to tensorboard
        print('Training loss: {:.4f}'.format(avg_loss))  # print loss to terminal

        model.eval()  # set eval mode
        total_loss = 0
        for batch_idx, (cell, pert, drug) in tqdm(enumerate(test_data_loader),  # iterate over test batches
                                                  total=len(test_data_loader),
                                                  desc=f'epoch {epoch}: Testing'):

            with torch.no_grad():
                c_ids, c_attn = test_data.tokenize_cells(cell)  # preprocess cells
                p_ids, p_attn = test_data.tokenize_cells(pert) # preprocess perturbations
                d_tok = test_data.tokenize_drugs(drug)  # preprocess drugs

                emb_pert, emb_drug = model({  # forward to model
                    'x': c_ids.to('cuda'),
                    'x_attn': c_attn.to('cuda'),
                    'y': p_ids.to('cuda'),
                    'y_attn': p_attn.to('cuda')
                }, d_tok.to('cuda'))

                total_loss += loss_fn(emb_drug, emb_pert ).item() # compute loss

            avg_loss = total_loss / len(test_data_loader)  # average loss
            writer.add_scalar('test_loss', avg_loss, epoch)  # print to tensorboard
            print('Test loss: {:.4f}'.format(avg_loss))  # log in terminal

            if avg_loss < min_eval_loss: # if improvement in validation save model
                best_epoch = epoch
                min_eval_loss = avg_loss
                torch.save(model.state_dict(), f'checkpoints/epoch_{epoch}.pt')


    # LOAD BEST MODEL
    model.load_state_dict(torch.load(f'checkpoints/epoch_{best_epoch}.pt'))

    print_embeddings_to_tensorboard(model,writer, test_data_loader, test_data)



def print_network_to_tensorboard(model, writer, test_data_loader, test_dataset):

    for batch_idx, (cell, pert, drug) in enumerate(test_data_loader):

        c_ids, c_attn = test_dataset.tokenize_cells(cell)  # preprocess cells
        p_ids, p_attn = test_dataset.tokenize_cells(pert)  # prerpcoess perturbations
        d_tok = test_dataset.tokenize_drugs(drug)  # prerpocess drugs

        writer.add_graph(model, ({
                                     'x': c_ids.to('cuda'),
                                     'x_attn': c_attn.to('cuda'),
                                     'y': p_ids.to('cuda'),
                                     'y_attn': p_attn.to('cuda')
                                 }, {

                                     'input_ids': d_tok['input_ids'].to('cuda'),
                                     'attention_mask' : d_tok['attention_mask'].to('cuda')

                                 }) )

        break



def print_embeddings_to_tensorboard(model, writer, test_data_loader, test_dataset):
    # EMBED VALID DATASET AND ADD TO TENSORBOARD EMBEDDING
    drug_embs = []
    pert_embs = []
    for batch_idx, (cell, pert, drug) in enumerate(test_data_loader):

        with torch.no_grad():
            c_ids, c_attn = test_dataset.tokenize_cells(cell)
            p_ids, p_attn = test_dataset.tokenize_cells(pert)
            d_tok = test_dataset.tokenize_drugs(drug)

            emb_pert, emb_drug = model({
                'x': c_ids.to('cuda'),
                'x_attn': c_attn.to('cuda'),
                'y': p_ids.to('cuda'),
                'y_attn': p_attn.to('cuda')
            }, d_tok.to('cuda'))

            drug_embs.append(emb_drug)
            pert_embs.append(emb_pert)

    # print embeddings to tensorboard
    # TODO: add drug name to both perturbation and drugs
    drug_embs = torch.concat(drug_embs)
    pert_embs = torch.concat(pert_embs)
    writer.add_embedding(
        torch.concat([drug_embs, pert_embs]),
        metadata = ['drug'] * len(drug_embs) + ['perturbation'] * len(pert_embs)
    )


