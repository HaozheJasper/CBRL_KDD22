import sys,os
import gym
sys.path.append(os.getcwd()+'/src')

from auction_simulator import Observation
from collections.abc import Sequence

from dqn import DQN
from sac import SAC, DDPG

import numpy as np
import pandas as pd
import time
from torch.utils.tensorboard import SummaryWriter
import logging
from runner_utils import make_parser, report, main, get_logger, train_or_test_synchronous_exp, Checkpointer, interfaces
import pickle as pkl
import random

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.getLogger('default').error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))  # 重点

sys.excepthook = handle_exception  # 重点


class BidAgent(interfaces):

    def get_logger_writer(self):
        self.logger = get_logger(self.output_dir)
        self.writer = SummaryWriter(self.output_dir)

    def _load_config(self, cfg):
        """
        Parse the config.cfg file
        """

        self.cfg = cfg
        # import ipdb; ipdb.set_trace()
        self.budget = eval(cfg['agent']['budget'])
        self.target_value = int(cfg['agent']['target_value'])
        self.T = int(cfg['rl_agent']['T'])  # T number of timesteps
        self.STATE_SIZE = 12 #int(cfg['rl_agent']['STATE_SIZE'])
        self.ACTION_SIZE = int(cfg['rl_agent']['ACTION_SIZE'])
        self.train_episodes_num = int(cfg['rl_agent']['train_num'])
        self.eval_freq = int(cfg['rl_agent']['eval_freq'])
        self.include_loosing = eval(cfg['data']['include_loosing'])
        self.use_syn = cfg['data']['use_syn']
        self.gamma = eval(cfg['rl_agent']['gamma'])
        self.discount = eval(cfg['rl_agent']['discount'])
        self.agent_type = cfg['rl_agent']['agent_type']
        self.ablation = eval(cfg['rl_agent']['ablation'])
        self.action_start = eval(self.cfg['rl_agent']['action_lo'])
        self.action_end = eval(self.cfg['rl_agent']['action_hi'])
        self.action_step = eval(self.cfg['rl_agent']['action_num'])
        self.max_loss_save = eval(self.cfg['rl_agent']['max_loss'])

        self.wkday_feature = eval(cfg['rl_agent']['wkday_feat'])
        self.future_costs_feature = eval(cfg['rl_agent']['future_costs'])

        self.restore_dir = cfg['rl_agent']['restore_dir']
        self.restore_epoch = cfg['rl_agent']['restore_epoch']
        use_history = eval(cfg['rl_agent']['use_history'])
        is_ep = eval(cfg['data']['gamma_nepoch'])>1 and eval(cfg['rl_agent']['rewardtype'])==6
        resume = eval(cfg['rl_agent']['resume'])
        self.is_ep = is_ep and use_history and resume

        self.slot_len_in_min = eval(cfg['data']['slot_len'])
        self.num_slot_in_period = eval(cfg['data']['period_len'])
        self.nslot_in_day = 24*60//self.slot_len_in_min

        self.test_incremental = eval(cfg['data']['test_inc'])
        self.ratio_update_type = eval(cfg['rl_agent']['ratio_update_type'])
        self.learn_limits = eval(cfg['rl_agent']['learn_limits'])

    def validate_action_space(self, env):
        max_oracle_ratio = env.max_oracle_ratio
        min_oracle_ratio = env.min_oracle_ratio
        min_action, max_action = self.action_start, self.action_end
        min_ratio,max_ratio = min_action, max_action
        span = max_ratio - min_ratio
        min_margin = min_oracle_ratio - min_ratio # should be positive
        max_margin = max_ratio - max_oracle_ratio # positive
        info = '{} - oracle ratio margin: min={}%({}vs{}),max={}%({}vs{})'.format(env.spec.id,
                                                                                 round(min_margin/span*100, 2),
                                                                                 min_ratio, min_oracle_ratio,
                                                                                 round(max_margin/span*100, 2),
                                                                                 max_ratio, max_oracle_ratio)
        if min_margin>=0 and max_margin>0:
            self.logger.info(info)
        else:
            raise Exception(info)

    def _get_state_size(self):
        pass
    def save_and_learn(self, env, can_train=False):
        pass
    def act(self, env, current_obs):
        pass
    # def batch_act(self):
    #     pass
    def eval(self):
        self.agent.eval()

    def set_train(self):
        self.agent.set_train()

    def get_multiplier(self):
        return self.agent.get_multiplier()

    def get_limit_parameter(self, detach=True):
        return self.agent.get_limit_parameter(detach=detach)

    def __init__(self, cfg, state_size, cat_sizes=tuple(), test_mode=False, envname='yewu', writer=None):
        self.test = test_mode
        self.envname = envname
        # self.BETA = self._get_action_space()  # [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
        self._load_config(cfg)
        self.BETA = self._get_action_space()  # [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
        buffertype = eval(cfg['rl_agent']['buffertype'])
        self.reward_type = eval(cfg['rl_agent']['rewardtype'])
        self.multi_timescale = self.reward_type==3 # use lagrangian

        self.penalty_scale = self.init_penalty_scale = float(cfg['hypers']['penalty'])  # float(cfg['agent']['penalty'])
        self.simplified = eval(cfg['rl_agent']['simplified'])
        self.model_name = '{}_slot({})_s({})_a({})'.format(self._get_model_name(), eval(cfg['data']['slot_len']),
                                                           state_size, len(self.BETA))

        self.output_dir = self.cfg['data'][
                              'output_dir'] + '/' + '{}_{}_{}_agent({})_dis({})_gamma({})_actiontype({})_reward({})_penalty({})_slot({})_wloose({})_syn({})_buffer({})'.format(
            time.strftime('%y%m%d%H%M%S', time.localtime()),
            self.model_name,
            self.envname,
            self.agent_type,
            self.discount,
            self.gamma,
            self.ratio_update_type,
            self.reward_type,
            self.penalty_scale,
            self.slot_len_in_min,
            self.include_loosing,
            self.use_syn,
            buffertype,

        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger_writer()
        if not self.test:
            self.logger.info('output to {}, train for {}'.format(self.output_dir, self.train_episodes_num))
            self.logger.info('using ep checkpointer: {}'.format(self.is_ep))
            self.cfg.write(open(os.path.join(self.output_dir, 'config.cfg'), 'w'))

        self.eps_start = 1.0
        self.eps_end = 0.05
        self.anneal = 0.00005

        # DQN Network to learn Q function
        replay_scheme = cfg['rl_agent']['replay_scheme']
        if self.agent_type=='dqn': # deprecated
            self.agent = DQN(cfg, state_size=state_size, action_size=len(self.BETA), seed=eval(cfg['data']['seed']), cat_sizes=cat_sizes,
                         use_wkday_feat=self.wkday_feature, buffertype=buffertype, replay_scheme=replay_scheme,
                         discount=self.discount, simplified=self.simplified)
        elif self.agent_type =='ddpg':
            action_bound = (self.action_start, self.action_end)

            self.agent = DDPG(cfg, state_size=state_size, action_size=1, seed=eval(cfg['data']['seed']),
                              action_bound=action_bound,
                              buffertype=buffertype, replay_scheme=replay_scheme,
                              discount=self.discount, simplified=self.simplified)
            self.logger.info('ddpg lr is {}, train freq is {}'.format(self.agent.lr, self.agent.train_freq))

        else:
            action_bound = (self.action_start, self.action_end)
            self.agent = SAC(cfg, state_size=state_size, action_size=1, seed=eval(cfg['data']['seed']), action_bound=action_bound,
                             buffertype=buffertype, replay_scheme=replay_scheme,
                         discount=self.discount, simplified=self.simplified)
            self.logger.info('sac lr is {}, train freq is {}'.format(self.agent.lr, self.agent.train_freq))


        self.logger.info('torch seed {}'.format(self.agent.torch_seed))
        self.buffertype = buffertype

        self.scheduler = self.agent.scheduler

        self.dqn_state = None
        self.dqn_action = 3  # no scaling
        self.dqn_reward = 0

        self.total_win = []
        self.total_rev = []
        self.total_cost = []
        self.total_bids = []
        self.total_win2 = []
        self.total_rev2 = []
        self.total_cost2 = []
        self.total_reward = [] # Fake
        self.total_reward_rw = [] # real

        self.qnet_loss, self.q_amp = None, None

        self.writer = SummaryWriter(self.output_dir)
        self.checkpointer = Checkpointer(is_ep=self.is_ep, old_yewu='Yewu' in self.envname)
        self.learn_step = 0

        dat = ['envname', 'actiontype', 'slotno', 'dayno', 'ratio', 'overall_roi', 'original_reward', 'normalized_reward', 'refratio',
               'step_winrate', 'step_rev', 'step_cost', 'step_roi', 'rev', 'cost', 'win', 'nbid'] + ['original_r', 'roi', 'roi-penalty', 'step_rev', 'rev-reward','exp-penalty']
        if eval(self.cfg['rl_agent']['rewardtype'])==3:
            dat.append('oracle_roi')
        # [original_r, roi, min(roi-1,0)*self.penalty_scale, step_rev, step_rev*self.reward_scale]
        pd.DataFrame([dat]).to_csv(os.path.join(self.output_dir, '{}_actions.csv'.format(self._get_model_name())),
                                   header=False,
                                   index=False, mode='w')

        self.init_lambda = eval(
            self.cfg['rl_agent']['init_lambda'])
        self.last_ratio = self.init_lambda

        self.eps = 1.

    def _get_action_space_independent(self):
        start = self.action_start #eval(self.cfg['rl_agent']['action_lo'])
        end = self.action_end # eval(self.cfg['rl_agent']['action_hi'])
        num_step = self.action_step # eval(self.cfg['rl_agent']['action_num'])
        step_size = (end-start)/num_step

        tmp = np.arange(start, end+step_size, step_size)
        return tmp

    def _get_action_space_correlated(self):
        return [-0.7,-0.5,-0.2,0.,0.2,0.5,0.7]

    def _get_action_space(self):
        if self.ratio_update_type==0:
            return self._get_action_space_independent()
        else: return self._get_action_space_correlated()

    def check_save(self):
        return self.checkpointer.check_save()

    def save(self, epoch, reward_scaler):
        self.agent.save(self.output_dir, '{}_{}'.format(self._get_model_name(), epoch), reward_scaler)
        # self.reward_net.save(self.output_dir)
    def load(self, epoch):
        print('scaler', self.agent.restore(self.output_dir, '{}_{}'.format(self._get_model_name(), epoch)))

    def clear_buffer(self):
        [x.clear() for x in self.agent.buffer.memory]

    def restore(self):
        return self.agent.restore(self.restore_dir, '{}_{}'.format(self._get_model_name(), self.restore_epoch))

    def set_period_exploration_factor(self, factor):
        self.eps = factor
        # print('exploration factor {} for next period'.format(self.eps))

    def set_exploration_factor(self, day_no):
        progress = (day_no % self.eval_freq) / self.eval_freq
        self.eps = self.eps_start * (1 - progress)
        print('exploration factor {} for next day {}'.format(self.eps, day_no))

    def update_total_stats(self, ): # todo use realworld
        self.total_win.append(self.wins_e_rw)
        self.total_rev.append(self.rev_e_rw)
        self.total_cost.append(self.cost_e_rw)
        self.total_bids.append(self.bids_e)
        self.total_win2.append(self.wins_e)
        self.total_rev2.append(self.rev_e)
        self.total_cost2.append(self.cost_e)
        self.total_reward.append(self.reward_e)
        self.total_reward_rw.append(self.reward_e_rw)

        self.logger.debug('compare fake/real:{}vs{},{}vs{},{}vs{}'.format(self.wins_e, self.wins_e_rw, self.rev_e, self.rev_e_rw, self.cost_e, self.cost_e_rw))

    def set_ratio(self, env, ratio):
        pass

class AgentWrapper(BidAgent):
    # naming heritage: discrete doesn't mean the agent has discrete action space
    def _get_model_name(self):
        return 'BaseAgent'

    def set_base_lambda(self, lamb):
        self.slot_lambda = lamb

    def _pv_step(self, env, current_obs):
        pass

    def update_buffer(self, tup, priority=None):
        if self.buffertype==4: self.update_buffer_per(tup, priority=priority)
        else: self.update_buffer_normal(tup, priority=priority)

    def clear_buffer(self):
        self.agent.clear_buffer()

    def update_buffer_normal(self, tup, priority=None):
        last_obs, action, r, obs, episode_done = tup
        self.agent.update_memory(None, last_obs, action, r, obs, episode_done)

    def update_buffer_per(self, tup, priority=None):
        self.agent.update_memory2(tup, priority)

    def dayover_logging(self):
        self.logger.debug('loss stats: q_amp={}, qnet_loss={}'.format(self.q_amp, self.qnet_loss))
        self.logger.debug('init value: {}'.format(self.agent.state_normalizer.init_value))
        # self.set_day(env.get_day())
        # self.update_total_stats()
    def checkpointer_add_acc(self, item):
        self.checkpointer.add_acc_item(item)

    def get_lr(self):
        return self.agent.get_lr()

    def scheduler_step(self):
        if self.scheduler is not None:
            if isinstance(self.scheduler, Sequence):
                [sched.step() for sched in self.scheduler]
            else: self.scheduler.step()
    def postprocess_reward(self, obs, action, nobs, rew):
        if rew is None: rew = 0
        return self.agent.postprocess_reward(obs, action, nobs, rew)

    def learn(self, can_train=False, iter_multiplier=1):
        if can_train:
            ret = self.agent.train(iter_multiplier=iter_multiplier)
            if self.multi_timescale and (self.learn_step+1)%(10//iter_multiplier)==0: # there will be 30x5x2 in total
                self.agent.train_multiplier()
                if (self.learn_step+1)%(100//iter_multiplier)==0: self.logger.info('Lagrangian multiplier {}'.format(self.get_multiplier()))

            qnet_loss, amp, qmin, qmax, qstd = ret['qnet']

            self.q_amp = amp
            if qnet_loss is not None:
                self.qnet_loss = qnet_loss
                self.checkpointer.add_loss_item(qnet_loss)
                self.writer.add_scalar('{}_stats/qnet_loss'.format('train'),
                                       qnet_loss, self.learn_step)
                self.writer.add_scalar('{}_stats/Q_targets'.format('train'),
                                       amp, self.learn_step)
                self.writer.add_scalar('{}_stats/Q_lr'.format('train'),
                                       self.get_lr(), self.learn_step)
                self.writer.add_scalar('{}_stats/Q_min'.format('train'),
                                       qmin, self.learn_step)
                self.writer.add_scalar('{}_stats/Q_max'.format('train'),
                                       qmax, self.learn_step)
                self.writer.add_scalar('{}_stats/Q_std'.format('train'),
                                       qstd, self.learn_step)
                if 'others' in ret:
                    aloss,ploss,entropy,kl_loss,mse_loss, cur_loss = ret['others']
                    if aloss is not None:
                        self.writer.add_scalar('{}_stats/policy_loss'.format('train'),
                                               ploss, self.learn_step)
                        self.writer.add_scalar('{}_stats/policy_entropy'.format('train'),
                                               entropy, self.learn_step)
                        self.writer.add_scalar('{}_stats/alpha_loss'.format('train'),
                                               aloss, self.learn_step)
                    self.writer.add_scalar('{}_stats/kl_loss'.format('train'),
                                           kl_loss, self.learn_step)
                    self.writer.add_scalar('{}_stats/mse_loss'.format('train'),
                                           mse_loss, self.learn_step)
                    self.writer.add_scalar('{}_stats/cur_loss'.format('train'),
                                           cur_loss, self.learn_step)
                self.learn_step += 1
                self.scheduler_step()
                # print(self.learn_step, self.get_lr())


    def batch_act(self, obs): # batch act before slot_step, not incremented
        if self.ratio_update_type==1:
            last_ratio = obs[1]
            allowed_idxes = [last_ratio+x>0 for x in self._get_action_space()]
        else: allowed_idxes = None

        action_next_slot = self.agent.act(obs, eps=self.eps if self.during_exp() else 1., is_deterministic=not self.during_exp(),
                                          allowed=allowed_idxes)
        return action_next_slot

    def _get_state_size(self):
        base_len = 10 if not self.simplified else 3
        if self.wkday_feature: base_len += 1
        if self.future_costs_feature: base_len += 4*2
        return base_len # temp

def train_before_test(is_sync, test_env, train_env, agent, model, test_day_unit, train_max_days, exploration_num, writer, logger, C, window_size, nepoch=1, gamma_nepoch=5, restore=False, do_random=True, init_b=0.2, save_margin=0.1):
    epoch_procedure = train_or_test_synchronous_exp
    agent.set_train()
    max_perf = -np.inf
    if train_env is not None:
        if train_env.reward_type==6:
            epochs = [5, 10]
            base_cnt = 0
            for git in range(gamma_nepoch):
                pb = init_b + 0.1 * git  # 0.8 enough
                train_env.set_b(pb)

                if gamma_nepoch == 1:
                    niter_each_epoch = nepoch
                elif gamma_nepoch == 2:
                    niter_each_epoch = 5 if git == 0 else 10
                # niter_each_epoch = nepoch if gamma_nepoch==1 else epochs[git]
                for it in range(niter_each_epoch):
                    if do_random: train_env.randomize_days()
                    max_perf = epoch_procedure(train_env, agent, logger, writer, window_size, exploration_num, C,
                                               istest=False, epoch_it=it + base_cnt, git=git, max_perf=max_perf,
                                               save_margin=save_margin)
                    train_env.set_step(0)

                agent.clear_buffer()
                base_cnt += niter_each_epoch
                # agent.load(0)
            train_env.save_mprice_model()
            test_env.set_b(pb)  # need to set gamma for test env, otherwise error occur
        else:
            for it in range(nepoch):
                print('epoch {}'.format(it))
                if do_random: train_env.randomize_days()
                max_perf = epoch_procedure(train_env, agent, window_size, exploration_num, C, istest=False, epoch_it=it,
                                           max_perf=max_perf, save_margin=save_margin)
                train_env.set_step(0)
            train_env.save_mprice_model()

    agent.eval()
    if not restore:
        agent.load(0)
    epoch_procedure(test_env, agent, logger, writer, window_size, exploration_num, C, istest=True, max_perf=max_perf)
    test_env.save(agent.output_dir)

def make_agent(cfg, test, envname):
    obs_size = Observation(cfg).size
    agent = AgentWrapper(cfg, obs_size, test_mode=test, envname=envname)  # BidAgent()
    return agent

if __name__ == '__main__':
    args = make_parser()
    main(args, make_agent, train_before_test)