import torch
from networkx import set_node_attributes
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx


def from_networkx_custom(G):
    """
    Converts a
    :obj:`networkx.Graph` or :obj:`networkx.DiGraph`
    to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    import networkx as nx

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            if (str(key) != "stereo"):
                data[str(key)] = [
                    value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data


class NetworkXTransform(object):

    def __init__(self, cat=False):

        self.cat = cat

    def __call__(self, data):

        x = data.x
        netx_data = to_networkx(data)
        ecc = self.nx_transform(netx_data)
        set_node_attributes(netx_data, ecc, 'x')
        ret_data = from_networkx(netx_data)
        ret_x = ret_data.x.view(-1, 1).type(torch.float32)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, ret_x], dim=-1)
        else:
            data.x = ret_x

        return data

    def nx_transform(self, networkx_data):
        """ returns a node dictionary with a single attribute
        """
        raise NotImplementedError
