import json
from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader,PedalMeDatasetLoader,WikiMathsDatasetLoader,WindmillOutputLargeDatasetLoader,WindmillOutputSmallDatasetLoader

class LocalEnglandCovidDatasetLoader(EnglandCovidDatasetLoader):
    def __init__(self):
        with open ('data\\england_covid.json' ,'r',encoding='utf8') as f:
            self._dataset = json.load(f)
class LocalPedalMeDatasetLoader(PedalMeDatasetLoader):
    def __init__(self):
        with open ('data\\pedalme_london.json' ,'r',encoding='utf8') as f:
            self._dataset = json.load(f)
class LocalWikiMathsDatasetLoader(WikiMathsDatasetLoader):
    def __init__(self):
        with open ('data\\wikivital_mathematics.json' ,'r',encoding='utf8') as f:
            self._dataset = json.load(f)
class LocalWindmillOutputLargeDatasetLoader(WindmillOutputLargeDatasetLoader):
    def __init__(self):
        with open ('data\\wikivital_mathematics.json' ,'r',encoding='utf8') as f:
            self._dataset = json.load(f)
class LocalWindmillOutputSmallDatasetLoader(WindmillOutputSmallDatasetLoader):
    def __init__(self):
        with open ('data\\windmill_output_small.json' ,'r',encoding='utf8') as f:
            self._dataset = json.load(f)            
if __name__=='__main__':
    loader = LocalEnglandCovidDatasetLoader()
    loader.get_dataset()