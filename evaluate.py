from train import PoolPredictor
import json
import os
import sys
from model import Sent2Vec
import numpy as np
from env import QAEnv
import pandas as pd
import torch

N_WORKERS = 40

np.random.seed(5005)
torch.manual_seed(5005)
torch.cuda.manual_seed_all(5005)



with open(sys.argv[1] + '.json') as f:
    config = json.load(f)['config']
train_env = QAEnv(config['data'], rules=config['rules'], evaluate=False)
dev_env = QAEnv(config['data'], rules=config['rules'], evaluate=True)
model = Sent2Vec(train_env, config)
model.load(config['model_path'])
model.eval()
pool = PoolPredictor(config['data'], N_WORKERS, config)
infos = pool.predict(model, list(range(len(dev_env))), use_dev=True)
df = {}
df['correct'] = [i['correct'] for i in infos]
df['rules'] = [i['rules'] for i in infos]
df['unifications'] = [i['unifications'] for i in infos]
df['scores'] = [i['scores'] for i in infos]
df['correct_idx'] = [i['correct_idx'] for i in infos]
df = pd.DataFrame(df)
df.to_csv(f'results/{os.path.basename(sys.argv[1])}_results.csv')

