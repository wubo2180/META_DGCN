import json
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader,EnglandCovidDatasetLoader,PedalMeDatasetLoader
class LocalChickenpoxDatasetLoader(ChickenpoxDatasetLoader):
    def __init__(self):
        with open ('dataset\\chickenpox.json' ,'r',encoding='utf8') as f:
            self._dataset = json.load(f)
            # self.lags = lags
            # self.preprocessing(lags)
    # def preprocessing(self, lags):
    #     dataset = self.get_dataset(lags)
    #     for time, snapshot in enumerate(dataset):
    #     # print(negative_sampling(snapshot.edge_index))
    #         pass
    
        # print(self._dataset)
if __name__=='__main__':
    loader = LocalChickenpoxDatasetLoader()
    loader.get_dataset()