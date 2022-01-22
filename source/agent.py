from abc import ABC, abstractmethod

from model import *


class Agent(ABC):
    def __init__(self, ndim, hsize):
        self.ndim = ndim
        self.hsize = hsize

    @abstractmethod
    def update_rates(self, denominator: float) -> None:
        """Linearly decay learning (and exploration) rates
        Args:
            denominator: Timescale of decay
        """


class QAgent(Agent, ABC):
    def __init__(self, env, ndim=10, hsize=4, agent_config=None):
        super().__init__(ndim, hsize)
        if agent_config is None:
            agent_config = {}
        agent_config.setdefault('embedding_dim', 12)
        agent_config.setdefault('action_space_size', hsize)
        agent_config.setdefault('policy_type', 'ActionIn')
        agent_config.setdefault('replay_capacity', int(1e5))
        agent_config.setdefault('learning_rate', 1e-4)
        agent_config.setdefault('batch_size', 500)
        agent_config.setdefault('eps_start', 1.)
        agent_config.setdefault('eps_end', 0.05)
        agent_config.setdefault('eps_decay', int(1e5))
        agent_config.setdefault('encoding_function', 'one_hot')
        agent_config.setdefault('num_head', 1)
        agent_config.setdefault('num_attn', 1)
        self.env = env
        self.agent_config = agent_config
        self.epsilon = agent_config['eps_start']
        self.steps_done = 0
        self.memory = ReplayMemory(self.agent_config['replay_capacity'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seq_len, embedding_dim = 2 * hsize + 1, self.agent_config['embedding_dim']

        # Encoding function: support one hot or sin wave
        if agent_config['encoding_function'] == 'one_hot':
            self.encoding_function = one_hot_encoding
        elif agent_config['encoding_function'] == 'sin':
            self.encoding_function = sin_positional_encoding
        else:
            raise ValueError("Invalid encoding function!")

        # Policy net: input tensor of size (B: batch size, N: sequence len, D: embed dim)
        # output tensor of size (B, num_actions) containing Q values
        if agent_config['policy_type'] == "ActionIn":
            self.policy_net = ActionInModel(seq_len, embedding_dim, num_head=agent_config['num_head'],
                                            num_attn=agent_config['num_attn'])
        elif agent_config['policy_type'] == "DQN":
            self.policy_net = DQNModel(seq_len, embedding_dim)
        else:
            raise ValueError("Given policy type invalid!")

        self.policy_net.to(self.device)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=agent_config['learning_rate'])

    def __str__(self) -> str:
        return self.agent_config['policy_type'] + '-' + self.agent_config['encoding_function'] + '-' + str(
            self.agent_config['num_attn']) + '-' + str(
            self.agent_config['num_head']) + '-' + str(self.env.ndim) + '-' + str(self.env.hsize)

    def detach_copy(self) -> Agent:
        """save agent model snapshot that only contains NN params but no memory"""
        new_agent = self.__class__(self.env, ndim=self.ndim, hsize=self.hsize, agent_config=self.agent_config)
        p = new_agent.policy_net.parameters()
        for par in p:
            print(par)
            break
        new_agent.policy_net.load_state_dict(self.policy_net.state_dict())
        new_agent.optimizer = torch.optim.AdamW(new_agent.policy_net.parameters(),
                                                lr=self.agent_config['learning_rate'])
        new_agent.optimizer.load_state_dict(self.optimizer.state_dict())
        new_agent.epsilon = self.epsilon
        new_agent.steps_done = self.steps_done
        return new_agent

    def update_rates(self, denominator: float) -> None:
        """update epsilon for agent's exploration"""
        self.epsilon = self.agent_config['eps_end'] + (
                self.agent_config['eps_start'] - self.agent_config['eps_end']) * \
                       math.exp(-1. * denominator / self.agent_config['eps_decay'])

    def transform_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """transform obs from array of integers to encodings"""
        # The obs is (h1, h2, target/hint), which are all 1D arrays
        return self.encoding_function(obs, d_model=self.agent_config[
            'embedding_dim'])  # (N,d), where N is the number of cards, d is the embedding dim

    def select_action(self, obs: torch.Tensor, evaluate=False) -> torch.Tensor:
        """given obs, select action for agent"""
        obs = self.transform_obs(obs)  # now obs is N*d
        obs = obs.unsqueeze(0)  # (1, N, D) for batch size = 1
        sample = random.random()
        if evaluate:
            self.policy_net.eval()
            with torch.no_grad():
                return self.policy_net(obs).max(1)[1].view(1, 1)
        self.steps_done += 1
        self.update_rates(self.steps_done)
        # print(sample, self.epsilon)
        if sample > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(obs).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.agent_config['action_space_size'])]], device=self.device,
                                dtype=torch.long)

    def optimize_model(self) -> (float, float, float):
        """one step gradient update"""
        if len(self.memory) < self.agent_config['batch_size']:
            return
        transitions = self.memory.sample(self.agent_config['batch_size'])
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state, dim=0)  # (bs, embdim, numcards)
        # print(state_batch.size())
        state_batch = torch.einsum('ben->bne', state_batch)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)
        expected_state_action_values = reward_batch
        criterion = nn.MSELoss()
        # criterion = nn.HuberLoss()
        loss = criterion(state_action_values.squeeze(1), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), expected_state_action_values.mean().item(), state_action_values.mean().item()
