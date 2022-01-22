import _pickle as cPickle
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml

from agent import QAgent
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

with open("../" + args.config_file, encoding='utf-8') as f:
    usr_config = yaml.safe_load(f)

print(torch.__version__)


def train_agents(verbose: bool = True, config: Optional[dict] = None):
    """train agents using configs and pickle results"""
    if config is None:
        config = {}
    # unpack config
    num_episodes = config.get("num_episodes", int(3e6))
    ndim, hsize = config.get("ndim", 5), config.get("hsize", 3)
    update_frequency = config.get("update_frequency", 250)
    agent_class = config.get("agent_class", QAgent)
    agent_config = config.get("agent_config", {})

    env = TinyHintGuessGame(ndim=ndim, hsize=hsize)
    hinter = agent_class(env, ndim=ndim, hsize=hsize, agent_config=agent_config)
    guesser = agent_class(env, ndim=ndim, hsize=hsize, agent_config=agent_config)

    hinter.policy_net.train()
    guesser.policy_net.train()

    running_rewards = []

    # calculate time hash for saving models and create str for agent
    print("Training started with configs:")
    print(config)
    hash_time = hashlib.sha1()
    hash_time.update(str(time.time()).encode('utf-8'))
    hash_time = hash_time.hexdigest()[:6]
    agent_str = str(hinter)

    # create path and save yaml file
    save_path = f"../{config['result_dir']}/{agent_str}-{hash_time}"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(f"{save_path}/configs.yaml", 'w') as handle:
        yaml.dump(config, handle)

    for i_episode in range(num_episodes):
        # run single episode
        obs_to_hinter = env.reset()
        hint_action = hinter.select_action(torch.tensor(obs_to_hinter, device=device))
        obs_to_guesser, _, _, _ = env.step(hint_action.item())
        guess_action = guesser.select_action(torch.tensor(obs_to_guesser, device=device))
        # print(hint_action, guess_action)
        _, r, _, _ = env.step(guess_action.item())

        # store the transition in memory
        obs_to_hinter = torch.tensor(np.array(obs_to_hinter), device=device)
        obs_to_guesser = torch.tensor(np.array(obs_to_guesser), device=device)
        hint_action = torch.tensor(np.array(hint_action), device=device)
        guess_action = torch.tensor(np.array(guess_action), device=device)
        r = torch.tensor([r], device=device)
        hinter.memory.push(hinter.transform_obs(obs_to_hinter).T, hint_action, None, r)
        guesser.memory.push(guesser.transform_obs(obs_to_guesser).T, guess_action, None, r)

        # perform one step of the optimization (on the policy network) every update_frequency steps
        if i_episode % update_frequency == 0:
            res1 = hinter.optimize_model()
            res2 = guesser.optimize_model()
            if res1:
                hloss, hq, hqhat = res1
                gloss, gq, gqhat = res2

        # append rewards to running rewards and print automatically
        running_rewards.append(r.cpu().numpy()[0])
        print_num = max(num_episodes // 500, 10)
        save_num = max(num_episodes // 20, 100)
        if i_episode > 1 and i_episode % print_num == 0:
            rw_to_print = np.array(running_rewards[-print_num:])
            running_rewards = []
            mean_win = np.sum(np.array(rw_to_print) >= 0, axis=0) / rw_to_print.shape[0]
            localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # logging and printing
            log_csv = f"{save_path}/log.csv"
            log_df = pd.DataFrame([[i_episode, mean_win, hloss, gloss, hq, hqhat, gq, gqhat, hinter.epsilon]],
                                  columns=['episodes', 'return', 'hinter_loss', 'guesser_loss',
                                           'hinter_q', 'hinter_qhat', 'guesser_q', 'guesser_qhat',
                                           'exploration'])
            log_csv_exists = os.path.isfile(log_csv)
            use_header = not log_csv_exists
            log_df.to_csv(log_csv,
                          index=False,
                          header=use_header,
                          mode='a',  # append data to csv file
                          chunksize=1)  # size of data to append for each loop
            if verbose:
                print(localtime, i_episode, mean_win, hinter.epsilon)
                print(round(hloss, 2), round(hq, 2), round(hqhat, 2), round(gloss, 2), round(gq, 2), round(gqhat, 2))

        # if i_episode > 1 and i_episode % save_num == 0:
        #     # save model snapshots periodically
        #     hinter_snapshot = hinter.detach_copy()
        #     guesser_snapshot = guesser.detach_copy()
        #     with open(f"{save_path}/{i_episode}.pkl", "wb") as output_file:
        #         cPickle.dump({'p1': hinter_snapshot, 'p2': guesser_snapshot}, output_file)
        #     print(f"Snapshot {i_episode} saved at {save_path}.")

    hinter.memory = None
    guesser.memory = None
    res = {'p1': hinter, 'p2': guesser}
    with open(f"{save_path}/{num_episodes}.pkl", "wb") as output_file:
        cPickle.dump({'p1': hinter, 'p2': guesser}, output_file)
    print(f"Final result saved at {save_path}.")
    return res


if __name__ == '__main__':
    res_dict = train_agents(config=usr_config, verbose=True)
    print(usr_config)
