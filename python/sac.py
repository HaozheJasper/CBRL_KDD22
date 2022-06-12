import numpy as np

import collections
from model import Network, StateActionValueNetwork, PolicyNetwork, StateNormalizer, DoubleQCritic, Critic, DoublePolicyNetwork, MyBuffer, NaiveBuffer

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import os
from torch.optim import lr_scheduler

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 24 * 2  # minibatch size
max_niter = 10
GAMMA = 1.0  # discount factor
TAU = 0.1  # for soft update of target parameters
LR = 0.001  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SAC():
    def __init__(self, cfg, state_size, action_size, seed, action_bound, buffertype=4,
                 replay_scheme='uniform', discount=1.0, simplified=False):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        torch.manual_seed(self.seed)
        # self.tau = 0.1
        # self.seed = random.seed(seed)
        self._load_config(cfg)

        # Q-Network
        if simplified:
            fc1_hidden = 20
        else:
            fc1_hidden = 30
        torch_seed = seed  # np.random.randint(10000)
        self.torch_seed = seed
        # networks: q; target q two nets respectively; one policy net
        self.sar_size = self.state_size + 2 if self.history_type=='sar' else 2*self.state_size+2
        if self.use_history:
            mlp_in = state_size + self.history_emb_dim
        else: mlp_in = state_size
        self.policy = self.get_actor(mlp_in, action_size, seed, action_bound, fc1_hidden)
        self.critic = self.get_critic(mlp_in, action_size, seed, fc1_hidden, self.tau)
        self.state_normalizer = StateNormalizer(cfg, state_size, use_bn=self.use_bn, value_in_state=self.ablation==16,
                                                use_history_encoder=self.use_history,
                                                encoder_type=self.encoder_type,
                                                history_emb_dim=self.history_emb_dim,
                                                history_size=self.history_size,
                                                history_type=self.history_type,
                                                use_history_decoder=self.use_decoder,
                                                reconstruction_num=self.reconstruction_num,
                                                use_curiosity=self.use_curiosity).to(device)

        self.networks = [self.critic, self.policy, self.state_normalizer]

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.optimizers = [self.policy_opt, self.critic_opt]

        if self.use_multiplier:
            log_mult = []
            if self.has_ROI_constraint:
                value = self.init_multiplier if self.init_multiplier>0 else np.random.uniform(0,4)
                tensor = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32).to(device), requires_grad=True)
            else: tensor = None
            log_mult.append(tensor)
            if self.has_budget:
                value = self.init_multiplier if self.init_multiplier>0 else np.random.uniform(0,4)
                tensor = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32).to(device), requires_grad=True)
            else: tensor = None
            log_mult.append(tensor)
            self.log_multiplier = log_mult
            self.multiplier_optimizer = torch.optim.Adam([x for x in log_mult if x is not None], lr=self.lr)
            # self.networks.append(self.log_multiplier)
            # self.optimizers.append(self.multiplier_optimizer) # don't participate in sched
        else: self.log_multiplier = None
        if self.learn_limits:
            self.limit_parameter = torch.nn.Parameter(torch.tensor(self.limit_init).to(device), requires_grad=True)
            self.limit_optimizer = torch.optim.Adam([self.limit_parameter], lr=self.lr*10)
            # while L is updated three days each, model parameters are update every time slot with at most 10 iters
            # self.optimizers.append(self.limit_optimizer) # don't participate in sched
        else: self.limit_parameter = None

        # [x.set_normalizer(self.state_normalizer) for x in self.networks]
        # self.networks.append(self.state_normalizer)
        params = list(self.state_normalizer.parameters())
        if len(params)>0:
            self.normalizer_opt = optim.Adam(params, lr=self.lr)
            self.optimizers.append(self.normalizer_opt)
        else: self.normalizer_opt = None

        if buffertype == 4:
            self._replay = self._build_replay_buffer()

        self.buffertype = buffertype
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self._replay_scheme = replay_scheme  # prioritized
        self.reward_discount = discount

        self.scheduler = self.get_sched()

    def get_critic(self, mlp_in, action_size, seed, fc1_hidden, tau):
        return DoubleQCritic(mlp_in, action_size, seed, fc1_hidden=fc1_hidden, use_bn=self.use_bn, tau=self.tau).to(device)

    def get_actor(self, mlp_in, action_size, seed, action_bound, fc1_hidden):
        return PolicyNetwork(mlp_in, action_size, seed, action_bound=action_bound, fc1_units=fc1_hidden,
                      use_bn=self.use_bn).to(device)

    def train_multiplier(self): # for RCPO: dual gradient descent
        if self.multi_timescale:
            self.multiplier_optimizer.zero_grad()
            for it in range(self._replay.num_episode_iter()):
                roi_diff,rbr = self._replay.sample_episode_batch()
                # min_alpha -alpha[totalcost - B] (totalcost <=B)
                # B-totalcost => remaining budget, so -remain_budget_rate \propto totalcost-B
                # so the surrogate loss is alpha x rbr
                # min_alpha -alpha[Lcost-rev] (ROI>=L => rev >= Lcost => Lcost-rev <=0)
                # L-rev/cost \propto Lcost-rev
                # so the surrogate loss is alpha x (ROI-L)
                device = self.get_device()
                multiplier_loss = torch.tensor(0.).to(device)
                if self.log_multiplier[0] is not None:
                    multiplier_loss += (self.log_multiplier[0] )*roi_diff.mean()
                if self.log_multiplier[1] is not None:
                    multiplier_loss += (self.log_multiplier[1] )*rbr.mean()
                multiplier_loss.backward()
            # nn.utils.clip_grad_norm_(self.log_multiplier, max_norm=self.clip_norm)
            self.multiplier_optimizer.step()


    def get_sched(self):
        sched = [lr_scheduler.MultiStepLR(opt, milestones=[4000, 8000, 12000], gamma=0.5) for opt in self.optimizers]
        return sched

    def get_niter(self):
        num_sample = self._replay.add_count
        max_batch = num_sample // self.batch_size
        return min(max_batch, max_niter)

    def eval(self):
        [x.eval() for x in self.networks]

    def set_train(self):
        [x.train() for x in self.networks]

    def _build_replay_buffer(self):
        """Creates the replay buffer used by the agent."""
        state_size = self.state_size if self.pe == 0 else self.state_size + 1
        extra_storage = []

        return NaiveBuffer(
                observation_shape=(state_size,),
                action_dtype=np.float32,
                stack_size=1,
                replay_capacity=self.buffer_size,
                batch_size=self.batch_size,
                update_horizon=1,
                gamma=1,
                observation_dtype=np.float32,
                extra_storage_types=extra_storage,
                require_nobs=True,
                use_history=self.use_history,
                history_size=self.history_size,
                history_type=self.history_type,
                sync=self.sync,
                horizon_length=self.num_slot_in_day
            )

    def clear_buffer(self):
        del self._replay
        self._replay = self._build_replay_buffer()

    def _load_config(self, cfg):
        self.cfg = cfg
        self.train_freq = eval(cfg['rl_agent']['train_freq'])  # in slots
        self.target_update_freq = eval(cfg['rl_agent']['target_freq'])  # default to 4 in slots
        self.buffer_size = eval(cfg['rl_agent']['buffer_size'])
        self.batch_size = eval(cfg['rl_agent']['batch_size'])
        self.min_history = self.batch_size
        self.lr = eval(cfg['rl_agent']['soft_q_lr'])
        self.pe = eval(cfg['rl_agent']['position_encoding'])
        self.use_bn = eval(cfg['rl_agent']['use_bn'])
        self.tau = eval(cfg['rl_agent']['tau'])
        self.ablation = eval(cfg['rl_agent']['ablation'])
        self.fc1_hidden = eval(cfg['rl_agent']['fc1_hidden'])
        self.use_history = eval(cfg['rl_agent']['use_history'])
        self.encoder_type = cfg['rl_agent']['encoder_type']
        self.history_size = eval(cfg['rl_agent']['history_size'])
        self.history_emb_dim = eval(cfg['rl_agent']['history_emb_dim'])
        self.history_type = cfg['rl_agent']['history_type']
        self.use_decoder = eval(cfg['rl_agent']['use_decoder'])
        self.stop_q_bp = eval(cfg['rl_agent']['stop_q_bp'])
        self.use_curiosity = eval(cfg['rl_agent']['use_curiosity'])
        self.curiosity_factor = eval(cfg['rl_agent']['curiosity_factor'])
        self.reconstruction_num = eval(cfg['rl_agent']['reconstruction_num'])
        slot_len = eval(cfg['data']['slot_len'])
        self.num_slot_in_day = (60 // slot_len) * 24
        self.sync = eval(cfg['data']['synchronous'])
        self.buffer_old = eval(cfg['rl_agent']['buffer_old'])
        self.clip_norm = eval(cfg['rl_agent']['clip_norm'])
        self.use_multiplier = eval(cfg['rl_agent']['rewardtype']) == 3
        self.init_multiplier = eval(cfg['rl_agent']['init_multiplier'])
        self.multi_timescale = self.use_multiplier and not eval(cfg['rl_agent']['tune_multiplier']) # not tune multiplier means learning
        self.learn_limits = eval(cfg['rl_agent']['learn_limits'])
        data_ver = cfg['data']['version']
        self.has_budget = 'b' in data_ver
        self.has_ROI_constraint = 'L' in data_ver or 'f' in data_ver
        self.limit_init = eval(cfg['rl_agent']['powerlaw_b'])


    def save(self, path, suffix, reward_scaler):
        names = ['critic', 'policy', 'state_normalizer']
        d = dict()

        for name,net in zip(names, self.networks):
            sd = net.state_dict()
            d[name] = sd
            # fpath = os.path.join(path, '{}_{}.pth'.format(name, suffix))
            # print('save to {}, '.format(fpath))
            # torch.save(sd, fpath)

        d['critic_opt'] = self.critic_opt.state_dict()
        d['policy_opt'] = self.policy_opt.state_dict()
        d['norm_opt'] = self.normalizer_opt.state_dict() if self.normalizer_opt else None
        d['reward_scaler'] = reward_scaler
        if self.use_multiplier:
            d['multiplier'] = self.log_multiplier
        if self.learn_limits:
            d['limits'] = self.limit_parameter

        fpath = os.path.join(path, '{}_{}.pth'.format('model', suffix))
        torch.save(d, fpath)

    def get_lr(self):
        return self.scheduler[0].get_lr()[0]

    def restore(self, path, suffix):
        names = ['critic', 'policy', 'state_normalizer']
        fpath = os.path.join(path, '{}_{}.pth'.format('model', suffix))
        if not os.path.exists(fpath):
            print('{} does not exist'.format(fpath))
            return None
        d = torch.load(fpath, map_location=torch.device('cpu'))
        for name, net in zip(names, self.networks):
            sd = d[name]
            # if not (name=='state_normalizer' and not self.use_bn):
            net.load_state_dict(sd, strict=False)
        # optimizer update lr
        for k in ['critic','policy']:
            d[k+'_opt']['param_groups'][0]['lr'] = self.get_lr()
        self.critic_opt.load_state_dict(d['critic_opt'])
        self.policy_opt.load_state_dict(d['policy_opt'])
        if self.use_multiplier:
            self.log_multiplier = [torch.nn.Parameter(torch.tensor(x.item()).to(self.get_device()), requires_grad=True) if x is not None else None for x in d['multiplier']]
            self.multiplier_optimizer = torch.optim.Adam([x for x in self.log_multiplier if x is not None], lr=self.get_lr())
        if self.learn_limits:
            self.limit_parameter = torch.nn.Parameter(d['limits'].to(self.get_device()), requires_grad=True)
            self.limit_optimizer = torch.optim.Adam([self.limit_parameter], lr=self.get_lr() * 10)
        if self.normalizer_opt:
            d['norm_opt']['param_groups'][0]['lr'] = self.get_lr()
            self.normalizer_opt.load_state_dict(d['norm_opt'])
        return d['reward_scaler']

    def update_memory2(self, tup, priority):
        last_obs, action, r, episode_done, obs = tup[:5]
        if len(tup)>5:
            epid_in_batch, cursor = tup[5:]
        else: epid_in_batch, cursor = None, None
        if priority is None:
            if self._replay_scheme == 'uniform':
                priority = 1.
            else:
                priority = self._replay.sum_tree.max_recorded_priority
        if self.use_history:
        #     init_arr = np.zeros(self.history_size*self.sar_size)
            last_obs, history, num_history = last_obs # remove the history
        #     obs,_,_ = obs
        if self.sync:
            self._replay.add(last_obs, action, r, episode_done)
        else:
            if self.buffer_old:
                self._replay.add(last_obs, action, r, episode_done)
            else:
                self._replay.add(last_obs, action, r, episode_done, epid_in_batch=epid_in_batch, cursor=cursor)

    def ok_to_sample(self):
        if hasattr(self._replay, 'ok_to_sample'): return self._replay.ok_to_sample()
        else:
            return self._replay.add_count > self.batch_size and self.t_step % self.train_freq == 0

    def _sample_from_replay_buffer(self):# todo: new subclass that overrides get transition elements
        samples = self._replay.sample_transition_batch()
        types = self._replay.get_transition_elements()
        self.replay_elements = collections.OrderedDict()
        for element, element_type in zip(samples, types):
            self.replay_elements[element_type.name] = element

    def compose_history(self, states, actions, next_states, rewards, histories, nums):
        """
        construct history for next states.
        histories are ordered in reverse order
        nums: list of ints
        """
        next_histories = np.empty_like(histories)
        if self.history_type=='sasr':
            toadd = np.concatenate([states, actions.reshape((-1,1)).cpu().numpy(), next_states, rewards.reshape(-1,1).cpu().numpy()], axis=-1) # num, n+2
        else:
            toadd = np.concatenate([states, actions.reshape((-1, 1)).cpu().numpy(), rewards.reshape(-1, 1).cpu().numpy()], axis=-1)  # num, n+2
        next_histories[:,:self.sar_size] = toadd # the newest is added to the head
        next_histories[:, self.sar_size:] = histories[:,:-self.sar_size]
        next_nums = np.clip(nums+1, 0, self.history_size)
        return next_histories, next_nums

    def train_agent(self, iter_multiplier=1):
        losses, amps, qmins, qmaxs, qstds, alosses, plosses, entropys, kls, mses, curs = [[] for _ in range(11)]
        self.t_step += 1

        num_iter = self.get_niter()
        mloss, mamp, mqmin, mqmax, mqstd,maloss,mploss,mentropy,mklloss,mmseloss, mcurloss = [None]*11
        if self.ok_to_sample():

            for _ in range(num_iter*iter_multiplier):

                self._sample_from_replay_buffer()
                if self._replay_scheme == 'prioritized':
                    # The original prioritized experience replay uses a linear exponent
                    # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of
                    # 0.5 on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders)
                    # suggested a fixed exponent actually performs better, except on Pong.
                    probs = self.replay_elements['sampling_probabilities']
                    # Weight the loss by the inverse priorities.
                    loss_weights = 1.0 / np.sqrt(probs + 1e-10)
                    loss_weights /= np.max(loss_weights)
                else:
                    loss_weights = np.ones(self.replay_elements['observation'].shape[0])

                states = self.replay_elements['observation'] # .squeeze(-1)
                actions = torch.from_numpy(self.replay_elements['action'])
                rewards = torch.from_numpy(self.replay_elements['reward'])
                next_states = self.replay_elements.get('next_obs', None)
                terminals = torch.from_numpy(self.replay_elements['terminal'])

                if self.use_history:
                    histories = self.replay_elements['history']
                    nums = self.replay_elements['num_history']

                    next_histories, next_nums = self.compose_history(states, actions, next_states, rewards, histories, nums)
                    experiences = ((states,histories, nums),
                                   actions, rewards,
                                   (next_states, next_histories, next_nums),
                                   terminals)
                else:
                    experiences = (states, actions, rewards, next_states, terminals)
                loss, amp, qmin, qmax, qstd, alpha_loss, policy_loss, entropy, kl_loss, mse_loss, cur_loss = self.learn(experiences, self.reward_discount, loss_weights)

                losses.append(loss)
                amps.append(amp)
                qmins.append(qmin)
                qmaxs.append(qmax)
                qstds.append(qstd)
                alosses.append(alpha_loss)
                plosses.append(policy_loss)
                entropys.append(entropy)
                kls.append(kl_loss)
                mses.append(mse_loss)
                curs.append(cur_loss)

            mloss, mamp, mqmin, mqmax, mqstd,maloss,mploss,mentropy = [torch.stack(x).mean().cpu().item() for x in
                                           [losses, amps, qmins, qmaxs, qstds,alosses,plosses,entropys]]
            if self.use_history:
                mklloss = torch.stack(kls).mean().cpu().item()
                if self.use_decoder: mmseloss = torch.stack(mses).mean().cpu().item()
                else: mmseloss = 0.
            else:
                mklloss = 0.
                mmseloss = 0.

            if self.use_curiosity:
                mcurloss = torch.stack(curs).mean().cpu().item()
            else: mcurloss = 0.

        if self.t_step % self.target_update_freq == 0:
            self.critic.sync_weights()
            self.policy.sync_weights() # can be ddpg
            # sync_weights(self.q1, self.target_q1, self.tau)
            # sync_weights(self.q2, self.target_q2, self.tau)

        return dict(qnet=[mloss, mamp, mqmin, mqmax, mqstd],others=[maloss,mploss,mentropy,mklloss, mmseloss, mcurloss])

    def train(self, iter_multiplier=1):
        ret = self.train_agent(iter_multiplier)
        return ret

    def get_multiplier(self):
        if self.use_multiplier:
            return [x.exp().item() if x is not None else None for x in self.log_multiplier]
        else: return [None, None]

    def get_limit_parameter(self, detach=True):
        if self.learn_limits:
            if detach: return max(self.limit_parameter.item(),0)
            else: return torch.relu(self.limit_parameter)
        else: return None

    def postprocess_reward(self, obs, action, nobs, rew): # for ICM reward
        if self.use_curiosity:
            device = self.policy.get_device()
            obs, action, nobs = [torch.FloatTensor([x]).to(device) for x in [obs, action, nobs]]
            diff, _, _ = self.state_normalizer.icm(obs, action, nobs)
            curiosity = self.curiosity_factor * diff.item()
            return rew + curiosity
        else:
            return rew

    def get_device(self): return self.policy.get_device()

    def act(self, state, eps=0., is_deterministic=False, allowed=None):
        flag = self.policy.training
        self.policy.eval()
        self.state_normalizer.eval()
        with torch.no_grad():
            # s, history, nhist = state
            ret = self.state_normalizer.forward(state, device=self.policy.get_device(), is_deterministic=is_deterministic)
            state = ret['state']

            if not is_deterministic:
                sampled_action, logp = self.policy.sample(state)
                ret = sampled_action
            else:
                _, scaled_mean, _, _, _ = self.policy.distr_params(state)
                ret = scaled_mean
        self.state_normalizer.train(flag)
        self.policy.train(flag)
        return ret

    def learn(self, experiences, gamma=1., loss_weights=None):
        states, actions, rewards, next_states, dones = experiences
        device = self.policy.get_device()
        actions, rewards, dones = actions.to(device), rewards.to(device), dones.to(device)
        num_sample = actions.shape[0]
        if not self.use_history:
            ret = self.state_normalizer.forward(np.concatenate([states,next_states], axis=0), device)
            tmp = ret['state']
            states, next_states = tmp[:num_sample], tmp[num_sample:]
            kl_loss = 0.
            mse_loss = 0.
        else:
            ret1 = self.state_normalizer.forward(states, device, kl_div=True, next_state_or_target_state=next_states if self.use_decoder else None)
            ret2 = self.state_normalizer.forward(next_states, device, kl_div=False)
            states, next_states = ret1['state'], ret2['state']
            kl_loss = ret1['kl_loss']
            if self.use_decoder: mse_loss = ret1['mse_loss']
            else: mse_loss = 0.

        if self.use_curiosity:
            curiosity_loss = self.state_normalizer.forward_curiosity_loss(states, actions, next_states)
        else: curiosity_loss = 0.
        new_action, new_action_logp, _, entropy = self.policy.sample(states.detach(), keep_params=True)
        next_action, next_action_logp = self.policy.sample(next_states)

        next_qv, target_qv = self.critic.target_qv(next_states, next_action, rewards, gamma, dones) # target_qv will be detached
        qloss, alpha_loss = self.critic.forward_loss(states.detach() if self.stop_q_bp else states, actions, new_action_logp, target_qv)

        if self.normalizer_opt: self.normalizer_opt.zero_grad()

        self.critic_opt.zero_grad()
        (alpha_loss+qloss+kl_loss+mse_loss+curiosity_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.clip_norm)
        if self.normalizer_opt: nn.utils.clip_grad_norm_(self.state_normalizer.parameters(), max_norm=self.clip_norm)
        self.critic_opt.step()
        if self.normalizer_opt: self.normalizer_opt.step()

        new_qv = self.critic.new_qv(states.detach(), new_action)
        alpha = self.critic.alpha
        # need to detach related states, if avg value in state
        policy_loss = (alpha * new_action_logp - new_qv).mean() # make policy close to q distribution

        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.clip_norm)
        self.policy_opt.step()

        return qloss, next_qv.mean(), next_qv.min(), next_qv.max(), next_qv.std(), alpha_loss, policy_loss, entropy.mean(), kl_loss, mse_loss, curiosity_loss

class DDPG(SAC):
    def learn(self, experiences, gamma=1., loss_weights=None):
        states, actions, rewards, next_states, dones = experiences
        device = self.policy.get_device()
        actions, rewards, dones = actions.to(device), rewards.to(device), dones.to(device)
        num_sample = actions.shape[0]
        zero_constant = torch.tensor(0.).to(device)
        if not self.use_history:
            ret = self.state_normalizer.forward(np.concatenate([states, next_states], axis=0), device)
            tmp = ret['state']
            states, next_states = tmp[:num_sample], tmp[num_sample:]
            kl_loss = zero_constant
            mse_loss = zero_constant
        else:
            ret1 = self.state_normalizer.forward(states, device, kl_div=True,
                                                 next_state_or_target_state=next_states if self.use_decoder else None)
            ret2 = self.state_normalizer.forward(next_states, device, kl_div=False)
            states, next_states = ret1['state'], ret2['state']
            kl_loss = ret1['kl_loss']
            if self.use_decoder:
                mse_loss = ret1['mse_loss']
            else:
                mse_loss = zero_constant

        if self.use_curiosity:
            curiosity_loss = self.state_normalizer.forward_curiosity_loss(states, actions, next_states)
        else:
            curiosity_loss = zero_constant

        next_action = self.policy.deterpolicy_target_action(next_states.detach(), is_deterministic=True)

        _, target_qv = self.critic.target_qv(next_states, next_action, rewards, gamma,
                                                   dones)  # target_qv will be detached
        qloss, alpha_loss = self.critic.forward_loss(states.detach() if self.stop_q_bp else states, actions,
                                                     None, target_qv)

        if self.normalizer_opt: self.normalizer_opt.zero_grad()

        self.critic_opt.zero_grad()
        (qloss + kl_loss + mse_loss + curiosity_loss).backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.clip_norm)
        if self.normalizer_opt: nn.utils.clip_grad_norm_(self.state_normalizer.parameters(), max_norm=self.clip_norm)
        self.critic_opt.step()
        if self.normalizer_opt: self.normalizer_opt.step()
        ########################
        new_action = self.policy.deterpolicy_action(states.detach(), is_deterministic=True)
        new_qv = self.critic.new_qv(states.detach(), new_action)
        # need to detach related states, if avg value in state
        policy_loss = (0 - new_qv).mean()  # make policy close to q distribution

        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.clip_norm)
        self.policy_opt.step()

        return qloss, new_qv.mean(), new_qv.min(), new_qv.max(), new_qv.std(), zero_constant, policy_loss, zero_constant, kl_loss, mse_loss, curiosity_loss

    def act(self, state, eps=0., is_deterministic=False, allowed=None):
        flag = self.policy.training
        self.policy.eval()
        self.state_normalizer.eval()
        with torch.no_grad():
            # s, history, nhist = state
            ret = self.state_normalizer.forward(state, device=self.policy.get_device(), is_deterministic=is_deterministic)
            state = ret['state']

            ret = self.policy.deterpolicy_action(state, is_deterministic=is_deterministic)
        self.state_normalizer.train(flag)
        self.policy.train(flag)
        return ret

    def get_critic(self, mlp_in, action_size, seed, fc1_hidden, tau):
        return Critic(mlp_in, action_size, seed, fc1_hidden=fc1_hidden, tau=tau).to(device)

    def get_actor(self, mlp_in, action_size, seed, action_bound, fc1_hidden):
        return DoublePolicyNetwork(mlp_in, action_size, seed, action_bound=action_bound, fc1_units=fc1_hidden,
                      use_bn=self.use_bn, tau=self.tau).to(device)



