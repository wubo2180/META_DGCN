# Enhancing Dynamic GCN for Node Attribute Forecasting with Meta Spatial-temporal Learning
This repository is the official implementation of paper Enhancing Dynamic GCN for Node Attribute Forecasting with Meta Spatial-temporal Learning

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset
Our datasets are EnglandCovid, PedalMe and WikiMaths.
All the necessary data files can be found at https://github.com/benedekrozemberczki/pytorch_geometric_temporal.


## Training and evaluation

To run our model, please run this command:

```shell
python main.py --dataset EnglandCovid  --update_sapce_step 1 --update_temporal_step 1 --k_spt 100 --k_qry 100 --device 0
```

