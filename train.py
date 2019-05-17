import json
import time
import os
from copy import deepcopy

import numpy as np
import ray
import torch
from tqdm import tqdm, trange
import sys

from env import QAEnv
from model import Sent2Vec

np.random.seed(5005)
torch.manual_seed(5005)
torch.cuda.manual_seed_all(5005)

BATCH_SIZE = 64
N_WORKERS = 40


def get_batches(idcs, batch_size):
    batches = []
    while True:
        try:
            batch_indices = np.random.choice(idcs, size=batch_size, replace=False)
            for idx in batch_indices:
                idcs.remove(idx)
            batches.append(batch_indices.tolist())
        except ValueError:
            return batches + [idcs]


@ray.remote
class Worker:
    def __init__(self, name, own_id, config):
        self.name = name
        self.train_env = QAEnv(name, rules=config['rules'])
        self.train_env.tnorm = 'prod'
        self.dev_env = QAEnv(name, rules=config['rules'], evaluate=True)
        self.dev_env.rules = self.train_env.rules
        self.dev_env._build_vocabs()
        self.dev_env.tnorm = 'prod'
        self.current_problem_id = None
        self.own_id = own_id


    def set_proof_params(self, min_bs_size, tnorm, lambda_cut, max_depth):
        self.train_env.min_bs_size = min_bs_size
        self.train_env.tnorm = tnorm
        self.train_env.lambda_cut = lambda_cut
        self.train_env.max_depth = max_depth

        self.dev_env.min_bs_size = min_bs_size
        self.dev_env.tnorm = tnorm
        self.dev_env.lambda_cut = lambda_cut
        self.dev_env.max_depth = max_depth

    def set_problem(self, i, dev=False):
        env = self.dev_env if dev else self.train_env
        obs = env.set_problem(i)
        self.current_problem_id = i
        return obs

    def act(self, sims, dev=False):
        env = self.dev_env if dev else self.train_env
        correct_idx = env.current_problem['candidates'].index(env.current_problem['answer'])

        info = env.step(sims)
        info['correct_idx'] = correct_idx
        info['pred_idx'] = np.argmax(info['scores'])
        info['correct'] = correct_idx == info['pred_idx']
        for idx in range(len(info['rules'])):
            unifications = env.current_program.get_unifications(info['rules'][idx], info['unifications'][idx])
            info['unifications'][idx] = unifications

        return self.current_problem_id, info, self.own_id


class PoolPredictor:
    def __init__(self, name, n_workers, config):
        ray.init(include_webui=False, local_mode=False, redirect_worker_output=False)
        self.workers = [Worker.remote(name, i, config) for i in range(n_workers)]

    def predict(self, model, problem_ids, use_dev=True, min_width=0, tnorm='prod', lambda_cut=0.5, max_depth=2):

        for worker in self.workers:
            worker.set_proof_params.remote(min_width, tnorm, lambda_cut, max_depth)

        all_ids = problem_ids
        problem_ids = deepcopy(problem_ids)
        infos = {}

        obs = [ray.get(worker.set_problem.remote(problem_id, use_dev)) for problem_id, worker in zip(problem_ids, self.workers)]
        sims = [model.get_sims(o) for o in obs]
        jobs = [worker.act.remote(sim, use_dev) for sim, worker in zip(sims, self.workers)]
        problem_ids = problem_ids[len(self.workers):]
        while len(problem_ids) + len(jobs):
            ready_ids, _ = ray.wait(jobs, num_returns=1)
            for ready_id in ready_ids:
                jobs.remove(ready_id)
                problem_id, result, worker_id = ray.get(ready_id)

                infos[problem_id] = result

                if len(problem_ids) > 0:
                    problem_id = problem_ids.pop(0)
                    obs = ray.get(self.workers[worker_id].set_problem.remote(problem_id, use_dev))
                    sims = model.get_sims(obs)
                    jobs.append(self.workers[worker_id].act.remote(sims, use_dev))

        infos = [infos[i] for i in all_ids]
        return infos


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        config = json.load(f)
        if len(config["model_path"]) == 0:
            config["model_path"] = "models/" + os.path.basename(sys.argv[1])

    train_env = QAEnv(config["data"], rules=config['rules'])
    train_ids = list(range(len(train_env.data)))
    dev_env = QAEnv(config["data"], rules=config['rules'], evaluate=True)
    dev_env.rules = train_env.rules
    dev_env._build_vocabs
    dev_ids = list(range(len(dev_env.data)))
    if config['type'].lower() == 'sent2vec':
        model = Sent2Vec(train_env, config)
    if config['reload']:
        model.load(config["model_path"])
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    pool = PoolPredictor(config["data"], N_WORKERS, config)


    step = 0
    best_dev_score = 0
    for epoch in range(config['epochs']):
        batches = get_batches(list(range(len(train_env))), BATCH_SIZE)


        model.eval()
        model.lambda_ = 1.0
        dev_infos = pool.predict(model, dev_ids, use_dev=True, tnorm='prod', min_width=0, lambda_cut=config["lambda_cut"], max_depth=3)
        dev_correct = [i['correct'] for i in dev_infos]
        dev_depths = [i['depths'][i['pred_idx']] for i in dev_infos]

        dev_acc = np.mean(dev_correct)

        print("dev acc:", dev_acc, "dev depth:", np.mean(dev_depths))
    
        if dev_acc > best_dev_score:
            best_dev_score = dev_acc
            model.save(config["model_path"])

        model.train()
        for batch in tqdm(batches):
            score = 0

            min_width = 0
            lambda_cut = config["lambda_cut"]

            infos = pool.predict(model, batch, use_dev=False, min_width=0, tnorm='prod', lambda_cut=lambda_cut, max_depth=3)

            train_correct = []
            train_depths = []
            n_losses = 0

            for info in infos:
                unifications = info['unifications'][info['correct_idx']]
                scores = deepcopy(info['scores'])
                scores[info['correct_idx']] = 0.0
                max_other_idx = np.argmax(scores)

                other_unifications = info['unifications'][max_other_idx]
                if len(unifications) == 0 == len(other_unifications):
                    print("No proof. Skipping...")
                    continue

                if len(unifications) > 0:
                    correct_score = model.recompute_score_with_grads(unifications)
                else:
                    correct_score = torch.tensor(0)

                if len(other_unifications) > 0:
                    other_score = model.recompute_score_with_grads(other_unifications)
                else:
                    other_score = torch.tensor(0)

                score += torch.log(correct_score) + torch.log(1-other_score)
                n_losses += 1

                train_correct.append(info['correct_idx'] == info['scores'].argmax())
                train_depths.append(info['depths'][info['pred_idx']])

            step += 1
            optim.zero_grad()
            try:
                loss = -score/n_losses
                loss.backward()
                optim.step()
            except AttributeError:
                print("Skipping model update, because of no proofs")


