import _pickle as cPickle
import argparse
import hashlib
import os
import sys
import time
from copy import deepcopy
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

print(torch.__version__)


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

    # calculate time hash for saving models
    print("Training started!")
    print(config)
    hash_time = hashlib.sha1()
    hash_time.update(str(time.time()).encode('utf-8'))
    hash_time = hash_time.hexdigest()[:8]
    agent_str = agent_config['policy_type'] + '-' + agent_config['encoding_function'] + '-' + str(ndim) + '-' + str(
        hsize)

    # create path and save yaml file)
    save_path = f"{config['result_dir']}/{agent_str}-{hash_time}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(f"{save_path}/configs.yaml", 'w') as handle:
        yaml.dump(config, handle)

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
        print_num = max(num_episodes // 500, 10)
        save_num = max(num_episodes // 10, 100)
        if i_episode > 1 and i_episode % print_num == 0:
            rw_to_print = np.array(rewards[-print_num:])
            rewards = []
            mean_win = np.sum(np.array(rw_to_print) >= 0, axis=0) / rw_to_print.shape[0]
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(f"{save_path}/train_log.txt", "a") as handle:
                handle.write(localtime + " Episodes: " + str(i_episode) + " Win rate: " + str(round(mean_win,3)) + " P1 loss: " + str(round(hloss, 3)) +
                             " P2 loss: " + str(round(gloss, 3)) + '\n')
                handle.write("P1 Q/Q_hat: " + str(round(hq, 2)) + "/" + str(round(hqhat, 2)) + " P2 Q/Q_hat " + str(
                    round(gq, 2)) + "/" + str(round(gqhat, 2)) + '\n')
            if verbose:
                print(localtime, i_episode, mean_win, hinter.epsilon)
                print(round(hloss, 2), round(hq, 2), round(hqhat, 2), round(gloss, 2), round(gq, 2), round(gqhat, 2))

        if i_episode > 1 and i_episode % save_num == 0:
            hinter_snapshot = deepcopy(hinter)
            guesser_snapshot = deepcopy(guesser)
            hinter_snapshot.memory = None
            guesser_snapshot.memory = None
            with open(f"{save_path}/{i_episode}.pkl", "wb") as output_file:
                cPickle.dump({'p1': hinter_snapshot, 'p2': guesser_snapshot}, output_file)
            print(f"Snapshot {i_episode} saved at {save_path}")

    hinter.memory = None
    guesser.memory = None
    res = {'p1': hinter, 'p2': guesser}
    return res


if __name__ == '__main__':
    print("Training started!")
    print(config)
    res_dict = train_agents(config=config, verbose=False)
    print(config)
