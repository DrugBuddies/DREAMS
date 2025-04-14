import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(model, loss_fn,
          train_data, test_data, collator,
          batch_size=4, lr=.001,
          n_epochs=200, device='cuda'):

    model.to(device)
    writer = SummaryWriter()

    # INITIALIZE OPTIMIZER AND DATA CLASSES

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, collate_fn=collator)

    #print_network_to_tensorboard(model, writer, data_loader, train_data)


    optimizer = Adam(model.parameters(), lr=lr)
    min_eval_loss = float('inf')
    best_epoch = 0
    for epoch in range(n_epochs):  # iterate over epochs
        model.train()  # set train mode
        total_loss = 0
        for batch_idx, (cells, drugs) in tqdm(enumerate(data_loader),  # iterate over train batches
                                                  total=len(data_loader),
                                                  desc=f'epoch {epoch}: Training'):


            optimizer.zero_grad()  # zero gradients


            emb_pert, emb_drug = model(cells, drugs)

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
        for batch_idx, (cells, drugs) in tqdm(enumerate(test_data_loader),  # iterate over test batches
                                                  total=len(test_data_loader),
                                                  desc=f'epoch {epoch}: Testing'):

            with torch.no_grad():
                emb_pert, emb_drug = model(cells, drugs)
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

    for batch_idx, (cell, drug) in enumerate(test_data_loader):

        writer.add_graph(model, (cell,drug) )

        break



def print_embeddings_to_tensorboard(model, writer, test_data_loader):
    # EMBED VALID DATASET AND ADD TO TENSORBOARD EMBEDDING
    drug_embs = []
    pert_embs = []
    for batch_idx, (cell, drug) in enumerate(test_data_loader):

        with torch.no_grad():
            emb_pert, emb_drug = model(cell,drug)
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


