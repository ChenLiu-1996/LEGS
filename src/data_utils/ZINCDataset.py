import logging

import numpy as np
import torch
from pysmiles import read_smiles
from torch.utils.data import Dataset
from torch_geometric.data import Data
from utils import from_networkx_custom

logging.getLogger('pysmiles').setLevel(logging.CRITICAL)


class ZINCDataset(Dataset):
    """
    ZINC Tranch data

    Returned `graph_data`: a torch_geometric.data.Data instance, where
                           graph_data.x is the node features and
                           graph_data.y is the properties.
    """

    def __init__(self, file_name, transform=None, prop_stat_dict=None, include_ki=False):

        self.prop_list = ['210409-Ikaros', '210312-ZNF410',
                          '210406-BCL11A', '210312-ZBTB7A', '210402-ZFP91']

        if include_ki:
            self.prop_list.append('Ki')

        self.tranch = np.load(file_name, allow_pickle=True).item()

        if prop_stat_dict != None:
            self.stats = np.load(prop_stat_dict, allow_pickle=True).item()
        else:
            self.stats = None

        self.transform = transform
        self.num_node_features = 10
        self.num_classes = len(self.prop_list)
        self.smi = list(self.tranch.keys())

    def __len__(self):

        return len(self.smi)

    def __getitem__(self, idx: int) -> Data:

        smi = self.smi[idx]

        properties = np.zeros(self.num_classes)
        no_zscore = np.zeros(self.num_classes)

        if self.stats != None:
            # We want to zscore.
            for i, entry in enumerate(self.prop_list):
                prop_value = self.tranch[smi][entry]
                z_scored = (
                    prop_value - self.stats[entry]['mean']) / self.stats[entry]['std']
                properties[i] = z_scored
        else:
            for i, entry in enumerate(self.prop_list):
                prop_value = self.tranch[smi][entry]
                properties[i] = prop_value
                no_zscore[i] = prop_value

        mol = read_smiles(smi)
        graph_data = from_networkx_custom(mol)

        graph_data.no_zscore_props = no_zscore
        graph_data.y = torch.Tensor(np.array([properties]))

        # Collect node features.
        node_features = []

        for i, entry in enumerate(graph_data.element):

            node_feature = np.zeros(self.num_node_features)

            # one hot encoding of atoms
            if entry == 'C':
                node_feature[0] = 1.
            elif entry == 'O':
                node_feature[1] = 1.
            elif entry == 'N':
                node_feature[2] = 1.
            elif entry == 'S':
                node_feature[3] = 1.

            # pair encoding of atoms
            if entry == 'C' or entry == 'O':
                node_feature[4] = 1.
            if entry == 'C' or entry == 'N':
                node_feature[5] = 1.
            if entry == 'C' or entry == 'S':
                node_feature[6] = 1.
            if entry == 'O' or entry == 'N':
                node_feature[7] = 1.
            if entry == 'O' or entry == 'S':
                node_feature[8] = 1.
            if entry == 'N' or entry == 'S':
                node_feature[9] = 1.

            node_features.append(node_feature)

        graph_data.x = torch.Tensor(np.array(node_features))

        if self.transform:
            return self.transform(graph_data)
        else:
            return graph_data
