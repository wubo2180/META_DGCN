from torch_geometric.utils import negative_sampling
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
def data_preprocessing(dataset):
    data_list = [] 
    for time, snapshot in enumerate(dataset):
        # print(negative_sampling(snapshot.edge_index))
        pass
    return data_list
    