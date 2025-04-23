from torch.utils.data import Dataset

class SMILESDataset(Dataset):
    def __init__(self, smiles, label):
        super(SMILESDataset, self).__init__()
        self.smiles = smiles
        self.label = label

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, index):
        smiles = self.smiles[index]
        label = self.label[index]
        return {'smiles': smiles, 'label': label}

    