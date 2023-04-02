import json
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader,EnglandCovidDatasetLoader,PedalMeDatasetLoader,WikiMathsDatasetLoader
class LocalChickenpoxDatasetLoader(ChickenpoxDatasetLoader):
    def __init__(self):
        with open ('dataset\\chickenpox.json' ,'r',encoding='utf8') as f:
            self._dataset = json.load(f)

class LocalPedalMeDatasetLoader(PedalMeDatasetLoader):
    def __init__(self):
        with open ('dataset\\pedalme_london.json' ,'r',encoding='utf8') as f:
            self._dataset = json.load(f)
            
class LocalEnglandCovidDatasetLoader(EnglandCovidDatasetLoader):
    def __init__(self):
        with open ('dataset\\england_covid.json' ,'r',encoding='utf8') as f:
            self._dataset = json.load(f)
            
class LocalWikiMathsDatasetLoader(WikiMathsDatasetLoader):
    def __init__(self):
        with open ('dataset\\wikivital_mathematics.json' ,'r',encoding='utf8') as f:
            self._dataset = json.load(f)
if __name__=='__main__':
    loader = LocalChickenpoxDatasetLoader()
    loader.get_dataset()