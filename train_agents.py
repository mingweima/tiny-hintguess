import _pickle as cPickle
import argparse
import datetime
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from agent import DQNAgent
from game import TinyHintGuessGame

sys.path.append(os.getcwd())
sys.path.append("")

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Specify the directory of the config file!')
parser.add_argument(
    "--config-file",
    default="configs/foo.yaml",
    metavar="FILE",
    help="path to config file",
    type=str,
)
args = parser.parse_args()

with open(args.config_file, encoding='utf-8') as f:
    config = yaml.safe_load(f)


def train_agents(verbose=True, config=None):
    if config is None:
        config = {}
    # unpack config
    num_episodes = config.get("num_episodes", int(3e6))
    ndim, hsize = config.get("ndim", 5), config.get("hsize", 3)
    update_frequency = config.get("update_frequency", 250)
    agent_class = config.get("agent_class", DQNAgent)
    agent_config = config.get("agent_config", {})

    env = TinyHintGuessGame(ndim=ndim, hsize=hsize)
    hinter = agent_class(env, ndim=ndim, hsize=hsize, agent_config=agent_config)
    guesser = agent_class(env, ndim=ndim, hsize=hsize, agent_config=agent_config)

    rewards = []
    for i_episode, _ in enumerate(tqdm(range(num_episodes))):
        obs_to_hinter = env.reset()
        hint_action = hinter.select_action(torch.tensor(obs_to_hinter, device=device))
        obs_to_guesser, _, _, _ = env.step(hint_action.item())
        guess_action = guesser.select_action(torch.tensor(obs_to_guesser, device=device))
        _, r, _, _ = env.step(guess_action)

        # Store the transition in memory
        obs_to_hinter = torch.tensor(np.array(obs_to_hinter), device=device)
        obs_to_guesser = torch.tensor(np.array(obs_to_guesser), device=device)
        hint_action = torch.tensor(np.array(hint_action), device=device)
        guess_action = torch.tensor(np.array(guess_action), device=device)
        r = torch.tensor([r], device=device)
        hinter.memory.push(hinter.transform_obs(obs_to_hinter).T, hint_action, None, r)
        guesser.memory.push(guesser.transform_obs(obs_to_guesser).T, guess_action, None, r)

        # Perform one step of the optimization (on the policy network)
        if i_episode % update_frequency == 0:
            res1 = hinter.optimize_model()
            res2 = guesser.optimize_model()
            if res1:
                hloss, hq, hqhat = res1
                gloss, gq, gqhat = res2

        rewards.append(r.cpu().numpy()[0])
        print_num = num_episodes // 100
        if verbose:
            if i_episode > 1000 and i_episode % print_num == 0:
                rw_to_print = np.array(rewards[-print_num:])
                rewards = []
                print(datetime.datetime.now(), i_episode,
                      np.sum(np.array(rw_to_print) >= 0, axis=0) / rw_to_print.shape[0], hinter.epsilon)
                print(round(hloss, 2), round(hq, 2), round(hqhat, 2), round(gloss, 2), round(gq, 2), round(gqhat, 2))

    hinter.memory = None
    guesser.memory = None
    resdict = {'p1': hinter, 'p2': guesser}
    return resdict


if __name__ == '__main__':
    print("Training started!")
    print(config)
    resdict = train_agents(config=config)
    hash_time = hashlib.sha1()
    hash_time.update(str(time.time()).encode('utf-8'))
    hash_time = hash_time.hexdigest()[:8]

    Path(f"{config['result_dir']}/{hash_time}").mkdir(parents=True, exist_ok=True)

    with open(f"{config['result_dir']}/{hash_time}/models.pkl", "wb") as output_file:
        cPickle.dump(resdict, output_file)
    with open(f"{config['result_dir']}/{hash_time}/configs.yaml", 'w') as f:
        yaml.dump(config, f)
    print(config)
    print(f"Results saved at {config['result_dir']}/{hash_time}")
