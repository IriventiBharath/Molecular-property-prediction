# import dependencies
import torch
import sklearn
import datasets
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
from MoLFormerWithRegressionHead import MoLFormerWithRegressionHead
from SMILESDataset import SMILESDataset
from sklearn.model_selection import train_test_split


DATASET_PATH = "scikit-fingerprints/MoleculeNet_Lipophilicity"
MODEL_NAME = "ibm/MoLFormer-XL-both-10pct"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
########################################################
# Entry point
########################################################

def Compute_gradients(dataloader, model, tokenizer, criterion):
    """
    Compute the gradients of the loss with respect to the model parameters for each batch in the dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of input samples.
        model (torch.nn.Module): The trained model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing SMILES strings.
        criterion (torch.nn.Module): Loss function used for training.

    Returns:
        list of lists of torch.Tensor: A list containing the computed gradients for each batch.
    """
    gradients = []
    model.eval()
    for i, batch in enumerate(dataloader):
        smiles = batch['smiles']
        smiles = tokenizer(smiles, padding=True, return_tensors='pt')
        label = batch['label'].float().reshape(len(batch['label']), 1)
        model.zero_grad()
        outputs = model(smiles['input_ids'], smiles['attention_mask'])
        loss = criterion(outputs, label)
        grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        gradients.append(grad)
    return gradients


def Test_gradient(batch, model, tokenizer, criterion):
    """
    Compute the gradients of the loss with respect to the model parameters for a single batch.
    """
    model.eval()
    smiles = batch['smiles']
    smiles = tokenizer(smiles, padding=True, return_tensors='pt')
    label = batch['label'].float().reshape(len(batch['label']), 1)
    model.zero_grad()
    outputs = model(smiles['input_ids'], smiles['attention_mask'])
    loss = criterion(outputs, label)
    grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    return [grad]


def Flatten_grad(grads):
    """
    Flatten a list of gradients into a single vector.

    Args:
        grads (list of torch.Tensor): List of gradients.

    Returns:
        torch.Tensor: Flattened gradients.
    """
    for i in range(len(grads)):
        flattened = []
        for g in grads[i]:
            flattened.append(torch.flatten(g))
        grads[i] = torch.cat(flattened)
    return grads



def Hvp(gradients, model, v):
    """
    Compute the Hessian vector product using the gradients of the loss with respect to the model parameters.

    Args:
        gradients (list of torch.Tensor): List of gradients.
        model (torch.nn.Module): The trained model.
        v (torch.Tensor): Vector to multiply with the Hessian.

    Returns:
        torch.Tensor: Hessian vector product.
    """
    HVP = torch.autograd.grad(outputs=gradients, inputs=model.parameters(), grad_outputs=v, retain_graph=True)
    HVP = torch.cat([torch.flatten(grad) for grad in HVP])
    return HVP


def Lissa_inverse_hessian_vector_product(gradients: list, model, T=10000, delta = 1e-2, scale = 10):
    """
    This function computes the inverse Hessian gradient product using the Lissa approximation:
    H^{-1}_0g = g
    H^{-1}_kv = g + (1 - \delta)*H^{-1}_{k-1}v - H*H^{-1}_{k-1}v
    INPUTS:
    gradients: list of gradients
    model: model nn
    OUTPUT:
    ihvp: list of inverse Hessian vector products
    """
    ihvp = []
    for g in gradients:
        Hv = g
        for i in range(T):
            HVP = Hvp(gradients=g, model=model, v=Hv)
            Hv = g + (1 - delta)*Hv - HVP/scale
            Hv = Hv/torch.norm(Hv)
        ihvp.append(Hv)
    return ihvp

def main():
    external_dataset = pd.read_csv('External-Dataset_for_Task2.csv') # external dataset
    # external_dataset = external_dataset.loc[n:n+3, :].reset_index()# take first n
    # print(external_dataset)
    external_dataset = SMILESDataset(external_dataset['SMILES'], external_dataset['Label']) # external dataset
    dataset = datasets.load_dataset(DATASET_PATH)
    train_dataset, test_dataset = train_test_split(SMILESDataset(dataset['train']['SMILES'], dataset['train']['label']), random_state=42) # same random state as the trained model
    dataset = {'train': train_dataset, 'test': test_dataset} # original dataset

    
    train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=16,shuffle=True) # training dataloader
    test_dataloader = torch.utils.data.DataLoader(dataset['test'], batch_size=16, shuffle=True) # test dataloader
    ed_dataloader = torch.utils.data.DataLoader(external_dataset, batch_size=1, shuffle=False) # external dataloader
    
    
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True) 
    model = MoLFormerWithRegressionHead(model)
    model.load_state_dict(torch.load('finetuned_regression_model.pt', weights_only=True))
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    criterion = torch.nn.MSELoss()
    
    
    gradients = Compute_gradients(dataloader=ed_dataloader, model=model, tokenizer=tokenizer, criterion=criterion) # gradients of the external dataset are 300
    gradients = Flatten_grad(gradients) # all gradients are vectors (44375809 x 1)^T 
    
    test_gradients = Compute_gradients(dataloader=test_dataloader, model=model, tokenizer=tokenizer, criterion=criterion) # gradients of train dataset are len(train_dataset)/Batch_size
    
    
    #test_gradient = Test_gradient(next(iter(test_dataloader)), model=model, tokenizer=tokenizer, criterion=criterion) # this takes one batch
   
   test_gradients = Flatten_grad(test_gradients)


    test_gradient = torch.mean(torch.stack(test_gradients), dim=0)

    ihvp = Lissa_inverse_hessian_vector_product(gradients, model=model, T=1000) # inverse Hessian vector product
    
    influence_scores = [-ihvp[i] @ test_gradient for i in range(len(ihvp))]
    
    # write results to a file

    with open('influence_scores_1.txt', 'a') as f:
        for score in influence_scores:

            f.write(str(score.item()) + '\n')



if __name__ == "__main__":
    main() 
