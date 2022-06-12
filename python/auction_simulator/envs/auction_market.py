"""
    Auction Emulator to generate bid requests from iPinYou DataSet.
"""
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
from matplotlib.image import imread
import configparser
import json
import os
import pandas as pd
import numpy as np
import time
import torch
# from
from tqdm import tqdm
from torchnet import meter
from multiprocessing import cpu_count
from glob import glob
import pickle as pkl
import random
import sys
sys.path.append('.')
from synthesize import YewuSampler2, find_oracle
from scipy.ndimage.filters import gaussian_filter1d
import abc
from typing import Sequence
from copy import copy
import itertools
import seaborn as sns
from collections import defaultdict

class Observation():
    def __init__(self, cfg):
        self.simplified = eval(cfg['rl_agent']['simplified'])
        self.ablation = eval(cfg['rl_agent']['ablation'])
        data_ver = cfg['data']['version']
        self.has_budget = 'b' in data_ver
        self.has_roi_constraints = 'L' in data_ver or 'f' in data_ver

        size = 3
        if self.has_budget>0: size += 1
        if not self.simplified:
            if self.ablation == -2: size += 2
            if self.ablation == -1:
                size += 1
            if self.ablation > 0:
                if self.ablation == 1:
                    size += 1
                elif self.ablation == 2:
                    size += 2
                elif self.ablation == 3:
                    size += 4
                elif self.ablation == 4:
                    size += 5
                elif self.ablation == 5:
                    size += 6
                elif self.ablation == 6:
                    size += 7
                elif self.ablation == 7:
                    size += 6
                elif self.ablation == 8:
                    size += 5
                elif self.ablation == 9:
                    size += 3
                elif self.ablation == 10:
                    size += 1
                elif self.ablation == 11:
                    size += 2
                elif self.ablation == 12:
                    size += 3
                elif self.ablation == 13:
                    size += 6
                elif self.ablation == 14:
                    size += 4
                elif self.ablation == 15:
                    size += 5
                elif self.ablation == 16:
                    size += 4
                elif self.ablation == 17:
                    size += 5
                elif self.ablation == 18:
                    size += 7
                elif self.ablation == 19:
                    size += 5
        if self.has_roi_constraints: size += 1
        self.size = size

class LimitedMeter():
    def __init__(self, size):
        self.memory = np.zeros(size, dtype=float)
        self.n = 0
        self.max_size = size
        self._iter = 0

    def add(self, sample):
        self.memory[self._iter % self.max_size] = sample
        if self.n < self.max_size:
            self.n += 1
        self._iter += 1

    def get_std(self):
        return self.memory[:self.n].std() if self.n>0 else 1.

    def get_mean(self):
        return self.memory[:self.n].mean() if self.n>0 else 0.

    def value(self):
        return self.get_mean(), self.get_std()

    def __len__(self):
        return self.n

class StateController():
    def __init__(self, nslot_in_day, cfg, budgets=None, constraints=None):
        """
        diverse budgets and constraints accept outer inputs
        """

        self.slot_scaler = 1. / nslot_in_day
        self.num_slot_in_day = nslot_in_day
        self.return_meter = meter.AverageValueMeter()
        self.penalty_meter = meter.AverageValueMeter()
        self.step_time_meter = meter.AverageValueMeter()
        # self.reward_meter = LimitedMeter(1000)
        # self.reward_meter = meter.AverageValueMeter()
        self.reward_meter = LimitedMeter(2000)
        self.reward_meter_rw = LimitedMeter(2000)
        self.avg_scale_records = []

        self.wkday_feature = eval(cfg['rl_agent']['wkday_feat'])
        self.future_costs_feature = eval(cfg['rl_agent']['future_costs'])
        self.simplified = eval(cfg['rl_agent']['simplified'])
        self.norm_with_mean = eval(cfg['rl_agent']['reward_norm_with_mean'])
        self.cem_nsample = eval(cfg['data']['cem_nsample'])
        self.cem_exp_percent = eval(cfg['data']['cem_exp_percent'])
        self.C = eval(cfg['data']['C'])
        self.cem_model = eval(cfg['cem']['model_type'])
        pid_ratio = eval(cfg['pid']['ratio'])
        self.init_ratio = eval(cfg['rl_agent']['init_lambda']) if pid_ratio==0 else pid_ratio
        self.ablation = eval(cfg['rl_agent']['ablation'])
        self.include_lost_data = eval(cfg['data']['include_loosing'])
        self.position_encoding = eval(cfg['rl_agent']['position_encoding'])
        self.use_history = eval(cfg['rl_agent']['use_history'])
        self.history_type = cfg['rl_agent']['history_type']
        self.history_size = eval(cfg['rl_agent']['history_size'])
        self.initial_budget = eval(cfg['data']['budget'])
        self.norm_reward = eval(cfg['rl_agent']['norm_reward'])

        self.observation = Observation(cfg)

        # self.last_action, self.last_ratio = None, self.init_ratio
        self._reset_day(0)
        self.preset_std, self.preset_rw_std = 0,0
        self.budgets = budgets
        self.has_budget = sum(self.budgets)!=0 if not isinstance(budgets, dict) else sum(budgets.values())# no budget: list of 0s
        self.roi_constraints = constraints
        self.has_roi_constraints = sum(constraints)!=0 if not isinstance(constraints, dict) else sum(constraints.values()) #  is not None
        self.remaining_budget = np.inf

    def get_day_budget(self, correct_dayno):
        if self.has_budget:
            return self.budgets[correct_dayno]
        else: return self.initial_budget

    def get_day_constraint(self, correct_dayno):
        return self.roi_constraints[correct_dayno] # only called when has roi constraints


    def set_reward_scaler(self, scaler):
        std, rw_std = scaler
        self.preset_std = std
        self.preset_rw_std = rw_std

    def get_cem_obs_size(self): return 1

    def get_cem_action_size(self):
        if self.cem_model==0: return 1
        else: return self.num_slot_in_day

    def get_basics(self):
        return [self.rev_t, self.cost_t, self.rev_t/(1e-4+self.cost_t), self.rev_e, self.cost_e, self.wins_e, self.bids_e]

    def get_overall_roi(self):
        return self.rev_e / (self.cost_e+1e-4)

    def get_overall_wr(self):
        return self.wins_e / (self.bids_e+1e-4)

    def get_step_wr(self):
        return self.wins_t / (self.bids_t+1e-4)

    def get_perf(self, past_day, mode_flag):
        roi = self.rev_e_rw / (self.cost_e_rw + 1e-4)
        # log
        perf = [mode_flag, past_day, past_day % 7,
                self.cost_e_rw, self.wins_e_rw, self.bids_e_rw,
                self.rev_e_rw,
                self.rev_e_rw / (self.cost_e_rw + 1e-4)]

        return perf

    def get_obs_size(self):
        return self.observation.size

    def get_history(self):
        """
        return: None or list of tuples
        """
        if self.use_history:
            if len(self.sar_list)>0:
                length = min(len(self.sar_list), self.history_size)
                return ([np.concatenate(x) for x in self.sar_list[-1:-length-1:-1]],length) # convenient to make next history if we use reversed order.
                # return (self.sar_list[-length - 1:-1], length) # in descending order, since latest info is important
            else: None
        else: return None

    def add_to_history(self, tup):
        if self.use_history:
            s,a,r = tup
            if self.history_type=='sar':
                self.sar_list.append((s[0], np.asarray([a.item(),r], dtype=np.float32))) # only the first part of obs is usable
            else:
                # todo: sasr
                pass
                # self.sar_list.append((s[0], ns, np.asarray([a.item(), r], dtype=np.float32)))

    def get_observation(self, extra):
        ratio = self.last_ratio
        dayno = extra['dayno']

        L = self.get_day_constraint(dayno)

        error = self.rev_e / (self.cost_e + 1e-4) - L if self.t_step>0 else 0
        must_have = [self.t_step * self.slot_scaler,  # hr, scale to [0,1]
                     ratio,
                     error]

        self.error_list.append(error)

        if self.has_roi_constraints: # todo: rearrange the position of this feature (to the end)
            must_have.append(self.get_day_constraint(dayno)) # if diverse constraint, should include the constraint target
        if not self.simplified:
            if self.ablation>0:
                ret = extra['revcost'] # last_rev, last_cost, this_rev, this_cost
                scaler = extra['scaler']

                real_roi = min(self.rev_t / (self.cost_t+1e-4) - self.get_day_constraint(dayno), 10)
                surplus = np.clip((self.rev_e-self.cost_e*L)*scaler, -50, 50)
                last_real_rev = min(self.rev_t * scaler, 50)

                must_have += [real_roi, last_real_rev, surplus]

        if self.position_encoding>0:
            if self.last_action is None:
                must_have.append(0)
            else: must_have.append(self.last_action+1)

        if self.t_step == 0:
            self.remaining_budget = self.get_day_budget(dayno)
            self.initial_budget = self.get_day_budget(dayno)

        if self.has_budget:
            budget_consumption = self.remaining_budget/self.initial_budget
            must_have.append(budget_consumption)

        tmp = np.asarray(must_have, dtype=np.float32)

        return tmp

    def _reset_slot(self):
        """
        Function to call every time a new time step is entered.
        """
        self.rev_t = 0.
        self.rev_t_rw = 0.
        self.cost_t = 0.
        self.wins_t = 0
        self.bids_t = 0
        self.last_step_ROI_e = self.rev_e / (self.cost_e + 1e-4)
        self.last_step_ROI_e_rw = self.rev_e_rw / (self.cost_e_rw+1e-4)

    def _reset_day(self, day_cnt=0):
        # super(BidAgent, self)._reset_episode(day_cnt)
        self.t_step = 0  # 1. t: the current time step

        self.WR = 0  # 6. wins_e / total_impressions
        self.cost_e = 0
        self.rev_e = 0
        self.wins_e = 0
        self.bids_e = 0
        self.cost_e_rw = 0
        self.rev_e_rw = 0
        self.wins_e_rw = 0
        self.bids_e_rw = 0
        self.exe_rev = 0
        self.exe_cost = 0
        self.last_step_value_avg = 0
        self.avg_value_sum = 0

        self._reset_slot()  # 7. Total value of the winning impressions 'click_prob'

        self.eps = 0. #self.eps_start
        self.daily_return = 0

        self.reward_e = 0
        self.reward_e_rw = 0

        self.last_action, self.last_ratio, self.error_list = None, self.init_ratio, []
        self.sar_list = []
        self.ratio_list = []
        # self.remaining_budget = self.budget if self.budget>0 else np.inf

    def _update_action_slot(self, action, ratio):
        self.last_action, self.last_ratio = action, ratio
        self.ratio_list.append(ratio)

    def _update_reward_slot(self, step_reward, step_reward_rw):
        self.reward_e += step_reward
        self.reward_e_rw += step_reward_rw

    def _update_exe_revcost_slot(self, t_step, rev, cost):
        self.exe_rev += rev
        self.exe_cost += cost

    def _update_rev_cost_slot(self, t_step, avg_value, po_rev, po_cost, po_nwins, po_nbids, fo_rev, fo_cost, fo_nwins, fo_nbids): # realtime update
        """
        po_revsum, po_costsum, po_nwins, po_nbids, fo_revsum, fo_costsum, fo_nwins, fo_nbids
        Internal function to update reward and action to compute the cumulative
        reward and cost within the given step.
        """
        # batch_fake_rew, batch_fake_cost, batch_fake_win, batch_nbids, batch_rew_t, batch_cost_t, batch_win_t
        # print('slot update', self.cost_t, self.cost_e, )
        self.rev_t = po_rev
        self.rev_t_rw = fo_rev
        self.cost_t = po_cost
        self.bids_t = po_nbids
        self.wins_t = po_nwins

        self.bids_e += po_nbids
        self.cost_e += po_cost
        self.rev_e += po_rev
        self.wins_e += po_nwins


        self.cost_e_rw += fo_cost
        self.rev_e_rw += fo_rev
        self.wins_e_rw += fo_nwins
        self.bids_e_rw += fo_nbids

        self.t_step = t_step
        self.last_step_value_avg = avg_value
        self.avg_value_sum += avg_value

        self.remaining_budget -= fo_cost

        # print('rev:{}, cost:{}'.format(reward/realworld_r, cost/realworld_c))


    # def _update_rev_cost(self, reward, cost, realworld_r, realworld_c): # realtime update
    #     """
    #     Internal function to update reward and action to compute the cumulative
    #     reward and cost within the given step.
    #     """
    #     self.rev_t += reward
    #     self.cost_t += cost
    #     self.bids_t += 1
    #     self.bids_e += 1
    #
    #     self.cost_e += (cost )
    #     self.rev_e += (reward )
    #     if cost > 0:
    #         self.wins_t += 1
    #         self.wins_e += 1
    #
    #     # self.cost_t_rw += realworld_c
    #     # self.rev_t_rw += realworld_r
    #     self.cost_e_rw += realworld_c
    #     self.rev_e_rw += realworld_r
    #     if realworld_c >0:
    #         # self.wins_t_rw += 1
    #         self.wins_e_rw += 1

    def stage(self):
        """
        store last step
        """
        self.storage = (self.t_step, self.rev_t, self.rev_t_rw, self.cost_t, self.wins_t, self.bids_t,
                        self.last_step_ROI_e, self.last_step_value_avg, self.rev_e, self.cost_e, self.wins_e, self.bids_e,
                        self.rev_e_rw, self.cost_e_rw, self.wins_e_rw, self.bids_e_rw, self.reward_e, self.reward_e_rw,
                        self.last_ratio, self.last_action, self.remaining_budget, copy(self.error_list), copy(self.sar_list))

    def stage_latest(self):
        self.latest = (self.t_step, self.rev_t, self.rev_t_rw, self.cost_t, self.wins_t, self.bids_t,
                        self.last_step_ROI_e, self.last_step_value_avg, self.rev_e, self.cost_e, self.wins_e, self.bids_e,
                       self.rev_e_rw, self.cost_e_rw, self.wins_e_rw, self.bids_e_rw, self.reward_e, self.reward_e_rw,
                       self.last_ratio, self.last_action, self.remaining_budget, copy(self.error_list), copy(self.sar_list))
        # print(self.bids_e)

    def recover(self):
        self.t_step, self.rev_t, self.rev_t_rw, self.cost_t, self.wins_t, self.bids_t, \
         self.last_step_ROI_e, self.last_step_value_avg, self.rev_e, self.cost_e, self.wins_e, self.bids_e,\
            self.rev_e_rw, self.cost_e_rw, self.wins_e_rw, self.bids_e_rw, self.reward_e, self.reward_e_rw,\
            self.last_ratio, self.last_action, self.remaining_budget, self.error_list, self.sar_list = self.latest

    def rollback(self):
        self.t_step, self.rev_t, self.rev_t_rw, self.cost_t, self.wins_t, self.bids_t,\
        self.last_step_ROI_e, self.last_step_value_avg, self.rev_e, self.cost_e, self.wins_e, self.bids_e, \
            self.rev_e_rw, self.cost_e_rw, self.wins_e_rw, self.bids_e_rw, self.reward_e, self.reward_e_rw,\
            self.last_ratio, self.last_action, self.remaining_budget, self.error_list, self.sar_list = self.storage

    def normalize_reward(self, step_reward, real=False):
        if self.norm_reward:
            if real:
                reward_meter = self.reward_meter_rw
                preset_std = self.preset_rw_std

            else:
                reward_meter = self.reward_meter
                preset_std = self.preset_std
            reward_meter.add(step_reward)
            if preset_std==0.: # means start from scratch
                tmp_std = preset_std if reward_meter.n<=self.num_slot_in_day else reward_meter.get_std()
            else:
                tmp_std = preset_std if reward_meter.n<=self.num_slot_in_day*5 else reward_meter.get_std()
            tmp_mean = 0.
            # tmp_mean = 0. if reward_meter.n<=self.num_slot_in_day*5 or not self.norm_with_mean else reward_meter.get_mean()
            rnet_r = (step_reward-tmp_mean) / (1. if tmp_std == 0. else tmp_std)  # todo: use limited average
            return rnet_r
        else: return step_reward

    def get_reward_std(self):
        return self.reward_meter.get_std(), self.reward_meter_rw.get_std()

class LoaderTemplate(metaclass=abc.ABCMeta):
    @abc.abstractmethod # 要求子类实现
    def get_fnames(self):
        pass

    @abc.abstractmethod
    def get_slot_oracle_ratio(self, dayno, slot_no):
        pass

    @abc.abstractmethod
    def get_rev_scaler(self, dayno):
        pass

    @abc.abstractmethod
    def get_wkday_base(self):
        pass

    @abc.abstractmethod
    def how_many_days(self):
        pass

    @abc.abstractmethod
    def get_day_data(self, day_no):
        pass

    def get_day_slot_data(self, dayno, slotno):
        curday_df = self.get_day_data(dayno)  # self.bid_requests_list[self.cur_day]

        selector = curday_df.index[curday_df.time_slot == slotno]
        has_rq = len(selector) > 0
        if has_rq: return has_rq, curday_df.loc[selector]
        else: return False, None

    @abc.abstractmethod
    def get_rev_by_dayslot(self, dayno, slotno):
        """
        fetch slot rev by dayno,slotno
        """
        pass
    @abc.abstractmethod
    def get_cost_by_dayslot(self, dayno, slotno):
        """
        fetch slot cost by dayno,slotno
        """
        pass

    @abc.abstractmethod
    def randomize_days(self):
        pass

    @abc.abstractmethod
    def get_slot_len(self) -> int:
        pass

    @abc.abstractmethod
    def get_oracle_arr(self, *args) -> Sequence:
        pass

    def update_df(self, df, idx, base=0, drop=0., minimal=False):
        """
        add missing parts to data
        params: base, in some loading methods, test set starts from `base`, but indexes starts from 0.
        return: df, dataframe; wins, bool array
        """
        slot_len = self.get_slot_len()
        slot_in_hour = 60 // slot_len

        df.loc[:, 'time_slot'] = df['hr'] * slot_in_hour + df['min'] // slot_len  if slot_len<60 else df['hr'] // (slot_len // 60 ) # df['day_no'] * slot_in_day +

        if 'pppc' not in df.columns:
            df = df.rename(columns=dict(ppc='pppc'))
        if 'click' not in df.columns:
            df = df.rename(columns=dict(clk1='click'))

        # if 'value' not in df.columns:
        #     df['value'] = df.pppc * df.pctr * 1e3
        df.loc[:, 'rev'] = df.rev.fillna(0.)

        if not minimal:
            oracles = self.get_oracle_arr(df, idx)
            df['oracle'] = oracles

        no_k2 = 'k2' not in df.columns
        if no_k2 and not minimal:
            k2_arr = self.get_k2_arr(df, base=base)
            df.loc[:,'k2'] = k2_arr
        wins = None
        if not minimal:
            if 'value' not in df.columns:
                df['value'] = df.pppc * df.pctr * 1e3
            bids = df.value * df.k2
            wins = bids > df.costprice
            print('k2 win rate {}'.format(wins.sum() / len(bids)))
            df['loose'] = True
            df.loc[wins, 'loose'] = False
        if drop>0:
            df.drop(index=np.random.choice(df.index, size=round(drop*df.shape[0])), inplace=True)
        return df, wins

    def get_k2_arr(self, df, base=0): # only for syn data
        k2_arr = []
        shape_df = df[['day_no', 'time_slot']]
        for (dn, ts), group in shape_df.groupby(['day_no', 'time_slot'], sort=False):
            dn -= base
            # 必须保证train/test连续
            k2_arr.append(np.asarray([self.get_slot_oracle_ratio(dn, ts)] * group.shape[0]))

        return np.concatenate(k2_arr)

    @abc.abstractmethod
    def get_day_fname(self, dayno):
        pass

    def set_logger(self, logger):
        self.logger = logger

# deprecated
class DataLoader(LoaderTemplate):
    def get_slot_len(self) -> int:
        return self.slot_len_in_min

    def get_data_fname(self, debug=False):
        return '../data/yewu-data/{}'.format('' if not debug else 'small_')+'{}_wloose_v4.csv'

    def get_slot_oracle_ratio(self, dayno, slot_no):
        return self.slot_oracles[dayno].get(slot_no, None)

    def get_oracle_arr(self, df, idx) -> Sequence:
        return df.day_no.map(lambda x: self.oracle_ratios[x])

    def preprocess(self, df, idx):
        df,wins = self.update_df(df, idx)
        # oracle rois

        tmp = df[wins]
        num_days = df.day_no.max() + 1
        num_slots = num_days*self.num_slot_in_day
        uprois = np.zeros(num_slots)

        self.cost_by_slot = []
        self.rev_by_slot = []
        self.oracle_ratios = np.zeros(num_days)
        revsum = None
        self.rev_scalers = []


        for dayno, dgroup in tmp.groupby('day_no'):
            uprev, upcost = 0., 0.
            cost_each_slot = np.zeros(self.num_slot_in_day)
            rev_each_slot = np.zeros(self.num_slot_in_day)
            for k,group in dgroup.groupby('time_slot'):
                cost_each_slot[k] = group.costprice.sum()
                rev_each_slot[k] = group.rev.sum()
            self.cost_by_slot.append(cost_each_slot)
            self.rev_by_slot.append(rev_each_slot)

            past_revsum = revsum
            revsum = dgroup.rev[~dgroup.loose].sum()
            if past_revsum is None: self.rev_scalers.append(1.)
            else: self.rev_scalers.append(past_revsum/revsum*self.rev_scalers[-1])

        return df

    def get_rev_scaler(self, dayno):
        return self.rev_scalers[dayno]

    def use_cols(self):
        often = ['wkday', 'hr', 'min', 'pid', 'pctr', 'pppc', 'value', 'costprice', 'rev',
                  'time_slot', 'loose', 'day_no']+list(self.reference_keys)
        if self.use_synthetic_cost!='none':
            if 'syn' in self.envname.lower():
                often += ['nick','ader','click']
            else: often += ['nick', 'click']

        return often

    def __init__(self, mode, debug, slot_len, refkeys, syn, envname, cols, data_folder):
        self.slot_len_in_min = slot_len
        self.num_slot_in_day = 60 // slot_len * 24
        self.reference_keys, self.use_synthetic_cost, self.envname, self.cols = refkeys, syn, envname, cols

        # fname = mode
        self.data_folder = data_folder
        read_fname = self.data_folder + '/data.csv'  # os.path.join(self.file_in, self.get_data_fname(self.debug).format(fname))
        st = time.time()
        df = pd.read_csv(read_fname)
        print('reading {} takes {}'.format(read_fname, time.time() - st))
        oracles = pkl.load(open(self.data_folder+'/oracles.pkl','rb'))

        self.oracle_ratios = oracles # list of ratios #[oracles.get(k, None) for k in range(max_key+1)]
        tmp = pkl.load(open(self.data_folder+'/oracles_s{}.pkl'.format(self.slot_len_in_min), 'rb'))
        self.slot_oracles = tmp # list of dict of slotid to ratios#[[] if k not in oracles else [tmp.get(k*self.num_slot_in_day+x, None) for x in range(self.num_slot_in_day)] for k in range(max_key+1)]
        df = self.preprocess(df)

        cols = self.use_cols()
        # df = df[df.day_no<2]
        df = df[cols]

        df.loc[:, 'agent'] = 0.

        self.max_oracle_ratio = df.oracle.max()
        self.min_oracle_ratio = df.oracle.min()
        self.bid_requests_list = [group for k, group in df.groupby('day_no')]
        self.df = df
        self.day_idxes = np.arange(self.how_many_days())

    def get_cats(self, df, cols):
        sizes = []
        for col in cols:
            if col in df:
                sizes.append(len(df[col].unique()))
        self.cat_sizes = sizes

    def get_meta(self):
        df = self.df
        self.max_oracle_ratio = df.oracle.max()
        self.min_oracle_ratio = df.oracle.min()

        self.all_bid_requests = df[['costprice','loose']].copy()
        # self.total_bids = df.shape[0]
        self.get_cats(df, self.cols)

        del df
        return self.oracle_ratios, self.cost_by_slot, self.max_oracle_ratio, self.min_oracle_ratio, self.all_bid_requests, self.cat_sizes

    def get_wkday_base(self):
        return self.bid_requests_list[0].wkday.iloc[0]

    def how_many_days(self):
        return len(self.bid_requests_list)

    def get_day_data(self, day_no):
        return self.bid_requests_list[day_no]

    def get_fnames(self):
        return self.day_idxes

    def get_day_fname(self, dayno):
        return self.day_idxes[dayno]

    def get_rev_by_dayslot(self, dayno, slotno):
        """
        fetch slot rev by dayno,slotno
        """
        return self.rev_by_slot[dayno][slotno] if self.num_slot_in_day>slotno>=0 else 0.

    def get_cost_by_dayslot(self, dayno, slotno):
        """
        fetch slot cost by dayno,slotno
        """
        return self.cost_by_slot[dayno][slotno]*1e-3 if self.num_slot_in_day>slotno>=0 else 0.

    def randomize_days(self):
        # day_idxes = np.arange(self.how_many_days())
        day_idxes = self.day_idxes = np.random.permutation(self.day_idxes)
        print('randomized days {}'.format(day_idxes))
        self.bid_requests_list = [self.bid_requests_list[i] for i in day_idxes]
        self.oracle_ratios = [self.oracle_ratios[i] for i in day_idxes]
        # self.slot_oracles = [self.slot_oracles[i] for i in day_idxes]
        self.rev_by_slot = [self.rev_by_slot[i] for i in day_idxes]
        self.cost_by_slot = [self.cost_by_slot[i] for i in day_idxes]

        revsum = None
        self.rev_scalers = []
        for group in self.bid_requests_list:
            past_revsum = revsum
            revsum = group.rev[~group.loose].sum()
            if past_revsum is None: self.rev_scalers.append(1.)
            else: self.rev_scalers.append(past_revsum/revsum*self.rev_scalers[-1])

# read logged dataset
class FileReader(LoaderTemplate):
    def __init__(self, folder, slot_len, cols, mode, train_days, train_files, is_cem=False, drop=0., setting='f', do_plot=False):
        self.data_folder = folder
        self.cols = cols
        self.drop = drop

        meta = pkl.load(open('../data/yewu-data/all0221/data_meta_all_f.pkl', 'rb'))

        self.meta = meta
        # self.policies = {k.split(os.path.sep)[-1].split('.')[0]: v for k,v in self.meta['policies'].items()}
        each_count = meta['each_count']
        self.file_paths = list(each_count.keys())#glob(self.data_folder+'/*.csv') # list of csv file paths
        if is_cem and train_days>=len(self.file_paths):
            raise Exception('train days {} not smaller than existing days {}'.format(train_days, len(self.file_paths)))

        num_slot_in_day = 24*60 // slot_len
        num_days = len(self.file_paths)
        self.istrain = mode=='train'
        self.train_days = train_days

        if train_files is None:
            self.file_paths = self.file_paths[:train_days] if mode=='train' else self.file_paths[train_days:]
        else:
            self.file_paths = train_files

        fpaths = [x.split(os.path.sep)[-1].split('.')[0] for x in self.file_paths]
        policies = meta['policies'][setting]
        k2_policy = meta['policies']['online']
        data_folder = os.path.sep.join(self.file_paths[0].split(os.path.sep)[:-1])
        if do_plot:
            self.slot_oracles = defaultdict(dict)
        else:

            dayid2soracle = {d.split(os.path.sep)[-1].split('.')[0]:p2f['slot_oracle']['ratios'] for d,p2f in policies.items()}
            self.slot_oracles = {data_folder+'/'+k+'.csv': v for k,v in dayid2soracle.items()} #[dayid2soracle[k] for k in fpaths]

        dayid2oracle = {k.split(os.path.sep)[-1].split('.')[0]: v['oracle']['ratio'] for k, v in policies.items()}
        self.oracle_ratios = [dayid2oracle[k] for k in fpaths]

        budgets = meta['budgets'][setting]
        self.budgets = {data_folder+'/'+k.split(os.path.sep)[-1]: v for k,v in budgets.items()}
        Ls = meta['Ls'][setting]
        self.Ls = {data_folder+'/'+k.split(os.path.sep)[-1]: v for k,v in Ls.items()}

        self.policies = dict() # {data_folder + '/' + k.split(os.path.sep)[-1]: v for k, v in policies.items()}
        for k,v in policies.items():
            nbid = k2_policy[k]['nbid']
            k2 = k2_policy[k]['k2']
            v['k2'] = k2
            v['nbid'] = nbid
            self.policies[data_folder + '/' + k.split(os.path.sep)[-1]] = v

        self.max_oracle_ratio = meta['max_ratio']
        self.min_oracle_ratio = meta['min_ratio']
        self.total_bids = meta['count']

        self.all_bid_requests = None  # df[['costprice','loose']].copy()
        # self.total_bids = df.shape[0]
        # pids, aders, users
        self.cat_sizes = [len(meta[k]) for k in self.cols]

        if len(self.file_paths)==0: raise Exception('folder might be incorrect: {}'.format(folder))
        self.slot_len_in_min = slot_len
        self.num_slot_in_day = 60 // slot_len * 24 if slot_len<60 else 24*60 // (slot_len)

        num_days = self.how_many_days()
        num_slots = num_days * self.num_slot_in_day
        self.oracle_rois = np.zeros(num_slots)

        self.revsum = None
        self.rev_scaler = 1.
        self.past_revsum = None
        # self.oracle_ratios = np.zeros(num_days)
        self.day_data, self.rev_by_slot, self.cost_by_slot = dict(), dict(), dict()
        self.ptr = -1

    def get_day_oracle(self, day):
        dayid = self.get_day_fname(day)

        policies = self.policies
        so_rev = policies[dayid]['slot_oracle']['rev']
        go_rev = policies[dayid]['oracle']['rev']
        return max(so_rev, go_rev)

    def get_slot_len(self) -> int:
        return self.slot_len_in_min

    def get_oracle_arr(self, df, idx) -> Sequence:
        return self.oracle_ratios[idx]

    def preprocess(self, df, idx):
        print('prepro {}'.format(self.file_paths[self.ptr]))
        df, wins = self.update_df(df, idx, base=self.train_days if not self.istrain else 0, drop=self.drop if self.istrain else 0.)

        tmp = df[wins]
        dayno = idx
        uprois = np.zeros(self.num_slot_in_day)
        # for dayno, dgroup in tmp.groupby('day_no'): # only single day
        uprev, upcost = 0., 0.

        tmp_cost_by_slot, tmp_rev_by_slot = np.zeros(self.num_slot_in_day), np.zeros(self.num_slot_in_day)
        for k,group in tmp.groupby('time_slot'):

            tmp_cost_by_slot[k] = group.costprice.sum()
            tmp_rev_by_slot[k] = group.rev.sum()

        self.rev_by_slot[dayno] = tmp_rev_by_slot
        self.cost_by_slot[dayno] = tmp_cost_by_slot

        print('prepro end')
        return df

    def get_rev_by_dayslot(self, dayno, slotno):
        """
        fetch slot rev by dayno,slotno
        """
        ans = 0.
        if dayno in self.rev_by_slot and self.num_slot_in_day > slotno >= 0:
            ans = self.rev_by_slot[dayno][slotno]
        return ans

    def get_day_rsum(self, dayno):
        return sum(self.rev_by_slot[dayno])

    def get_cost_by_dayslot(self, dayno, slotno):
        """
        fetch slot cost by dayno,slotno
        """
        ans = 0.
        if dayno in self.cost_by_slot and self.num_slot_in_day > slotno >= 0:
            ans = self.cost_by_slot[dayno][slotno]
        return ans*1e-3

    def get_slot_oracle_ratio(self, dayno, slot_no):
        fname = self.get_day_fname(dayno) # input normal day idx
        return self.slot_oracles[fname].get(slot_no, None)

    def get_meta(self):
        # load meta of dataset: max/min oracle, total bids, cat_sizes, costprice/loose histogram
        return self.oracle_ratios, self.cost_by_slot, self.max_oracle_ratio, self.min_oracle_ratio, self.all_bid_requests, self.cat_sizes

    def read_data(self, dayno):
        if dayno>=self.how_many_days():
            df = self.day_data[dayno-1]
        else:
            df = pd.read_csv(self.file_paths[dayno]) # 是一个trick，在最后一天不能读下一天，于是用当天冒充了，反正不会真的用到
            df = self.preprocess(df, dayno)
        self.day_data[dayno] = df
        tmp = self.revsum
        self.revsum = df.rev[~df.loose].sum()
        if self.past_revsum is not None:
            self.rev_scaler *= self.past_revsum / self.revsum # reciprocal
        self.past_revsum = tmp

    def get_day_data(self, day_no, drop=None):
        if day_no not in self.day_data:
            # print('removing rank-{}, reading rank-{}'.format(self.ptr, day_no))
            self.ptr = day_no
            self.read_data(day_no)

        if drop in self.day_data:
            self.day_data.pop(drop)
            self.rev_by_slot.pop(drop)
            self.cost_by_slot.pop(drop)

        return self.day_data[day_no]

    def get_fnames(self):
        return self.file_paths

    def get_day_fname(self, dayno):
        return self.file_paths[dayno]

    def get_rev_scaler(self, day_no):
        # return self.oracle_revs[self.get_day_fname(day_no)]
        return 48./self.get_day_oracle(day_no)

    def randomize_days(self):
        day_idxes = np.arange(self.how_many_days())
        day_idxes = np.random.permutation(day_idxes)
        self.day_idxes = day_idxes

        self.file_paths = [self.file_paths[i] for i in day_idxes]
        self.oracle_ratios = [self.oracle_ratios[i] for i in day_idxes]
        # self.slot_oracles = [self.slot_oracles[i] for i in day_idxes]
        print('randomized days {}'.format(self.file_paths))

    # def reset(self):
    #     self.ptr = 0
    #     self.read_data()

    def get_wkday_base(self):
        return 0 # self.day_data[0].wkday.iloc[0]

    def how_many_days(self):
        return len(self.file_paths)

# generate slot-wise synthetic data
class SlotGenerator(LoaderTemplate):
    def __init__(self, slot_len, num_days, num_sample_per_day, iid, data_seeds, gen_meta=None, data_ver=None, Ls=tuple(), budgets=tuple()):
        super(SlotGenerator, self).__init__()
        self.Ls = Ls # C can be list
        self.has_roi_constraints = sum(Ls)!=0 # isinstance(Ls, list)
        self.budgets = budgets
        self.has_budget = sum(budgets)!=0

        self.max_oracle_ratio = 3.0 #meta['max_ratio']
        self.min_oracle_ratio = 1.0 #meta['min_ratio']
        # self.exp_seed = exp_seed
        self.cat_sizes = [] #[len(meta[k]) for k in self.cols]

        self.slot_len_in_min = slot_len
        self.num_slot_in_day = 60 // slot_len * 24
        self.num_days = num_days
        num_slots = num_days * self.num_slot_in_day
        # self.oracle_rois = np.zeros(num_slots)
        self.cost_by_slot = np.zeros(num_slots)
        self.revsum = None
        self.past_revsum = None
        self.rev_scaler = 1.
        # self.oracle_ratios = np.zeros(num_days)
        self.num_sample_per_day = num_sample_per_day
        self.data_ver = data_ver
        self.sampler = YewuSampler2(0, False, num_days, iid, data_ver=data_ver)
        self.fixed_data_seeds = data_seeds
        self.amps, self.noises = self.sampler.get_shifts(num_days, slot_len) # set seed == 0
        self.roi_high_hours = self.sampler.get_high_roi_hours(num_days)
        self.sampler.cost_planning(self.amps, self.noises, self.roi_high_hours, self.slot_len_in_min, self.num_sample_per_day)
        # np.random.seed(exp_seed)
        # random.seed(exp_seed)
        self.day_idxes = np.arange(self.how_many_days())

        self.cost_by_slot = None
        self.ptr = -1
        self.gen_meta = gen_meta
        self.oracle_ratios = gen_meta['day_stats']['rev_oracle']['ratio'] if gen_meta is not None else None
        # line 1500
        # day_stats = dict(slot_oracle=dict(rev=[],cost=[],nwin=[]), nbid=[],
        #                      value_oracle=dict(rev=[],cost=[],nwin=[],ratio=[]),
        #                      rev_oracle=dict(rev=[], cost=[], nwin=[], ratio=[]),
        #                      avg_value=[],
        #                      dynamics=[],
        #                      corresponding_yewu=generator.sampler.selected_days
        #                      )
        self.require_online_oracle_search = self.gen_meta is None  # online generate oracle info
        # self.init_day_data(0)
        self.day_data = dict()
        self.slot_oracles = dict()
        self.slot_value_sum = dict()
        # self.oracle_revs = None if self.gen_meta is None else self.gen_meta['day_stats']['rev_oracle']['rev']
        self.dynamics = dict()
        keys = ['costprice', 'rev', 'value', 'pppc']
        self.keys = keys
        for key in keys:
            self.dynamics[key+'_mean'] = dict()
        self.dynamics['ctr'] = dict()
        # self.gen_meta = gen_meta

    def get_day_oracle(self, day):
        dayno = self.get_day_fname(day)
        policies = self.gen_meta['day_stats']
        so_rev = policies['slot_oracle']['rev'][dayno]
        go_rev = policies['rev_oracle']['rev'][dayno]
        final_rev = max(so_rev, go_rev)

        return final_rev

    def init_day_data(self, dayno):
        self.day_data = dict()
        self.rev_by_slot = dict()
        self.slotoracle_rev_by_slot = dict()
        self.slotoracle_cost_by_slot = dict()
        self.cost_by_slot = dict()
        self.win_by_slot = dict()
        self.nbid_by_slot = dict()
        self.slot_oracles = dict()
        if not self.require_online_oracle_search:
            dayslot_meta = self.gen_meta['day_slot_meta']
            self.rev_by_slot = dayslot_meta['rev'][dayno]
            self.cost_by_slot = dayslot_meta['cost'][dayno]
            self.nwin_by_slot = dayslot_meta['win'][dayno]
            self.nbid_by_slot = dayslot_meta['nbid'][dayno]
            self.slot_oracles = dayslot_meta['slot_ratios'][dayno]
        self.oracle_revcost_by_slot_thrsh = [] # slot, thrsh, 2

    def set_logger(self, logger):
        super(SlotGenerator, self).set_logger(logger)
        self.sampler.set_logger(logger)

    def get_slot_oracle_ratio(self, dayno, slot_no):
        return self.slot_oracles.get(slot_no, None)

    def get_slot_len(self) -> int:
        return self.slot_len_in_min

    def get_oracle_arr(self, df, idx):
        return self.oracle_ratios

    def get_rev_by_dayslot(self, dayno, slotno):
        """
        fetch slot rev by dayno,slotno
        """
        if slotno not in self.rev_by_slot:
            self.get_day_slot_data(dayno, slotno)
        return self.rev_by_slot[slotno]

    def get_day_rsum(self, dayno):
        return sum(self.rev_by_slot.values())

    def get_cost_by_dayslot(self, dayno, slotno):
        """
        fetch slot cost by dayno,slotno
        """
        return self.cost_by_slot[slotno]

    def get_meta(self):
            # load meta of dataset: max/min oracle, total bids, cat_sizes, costprice/loose histogram
            return self.oracle_ratios, self.cost_by_slot, self.max_oracle_ratio, self.min_oracle_ratio, None, self.cat_sizes

    def stats_of_day(self):
        revsum = sum(self.slotoracle_rev_by_slot.values())
        costsum = sum(self.slotoracle_cost_by_slot.values())
        winsum = sum(self.win_by_slot.values())
        bidsum = sum(self.nbid_by_slot.values())

        return revsum, costsum, winsum, bidsum

    def get_day_slot_data(self, dayno, slotno, drop=None, threshes=None, do_plot=False):
        # idx before randomization
        if threshes is None:
            threshes = np.arange(0.01, 5., 0.01)

        dayno = self.get_day_fname(dayno)
        if drop in self.day_data:
            del self.day_data[drop]

        if slotno not in self.day_data:
            if slotno == 0:  # not sure if this happens after logging at the day end
                self.init_day_data(dayno)
                seed = self.fixed_data_seeds[dayno]
                self.sampler.set_seed(seed)
            # st = time.time()
            data = self.sampler.sample_slot(dayno, slotno, do_plot)
            # print('sample slot, ', time.time()-st)
            # st = time.time()
            df, _ = self.update_df(data, dayno, minimal=True)
            self.day_data[slotno] = df

        return True, self.day_data[slotno]

    def matrix_find_oracle(self, values, costs, revs):
        """
        values costs same scale, should normalized by 1e-3
        revs normal scale
        """
        threshes = np.arange(0.5, 5., 0.025)
        bids = np.outer(threshes, values)  # m,n

        wins = (bids > costs[None, :])
        winsum = wins.sum(1)
        won_rsum = np.einsum('mn,n->m', wins, revs)
        won_csum = np.einsum('mn,n->m', wins, costs) * 1e-3
        # tmp = np.stack([won_rsum, won_csum, winsum], axis=1)
        won_roi = won_rsum/won_csum
        bool_arr = won_roi>1.
        arg_ind = won_rsum[bool_arr].argmax()
        ans_th = threshes[bool_arr][arg_ind]
        ans_r = won_rsum[bool_arr][arg_ind]
        ans_c = won_csum[bool_arr][arg_ind]
        ans_w = winsum[bool_arr][arg_ind]
        nbid = values.shape[0]
        return ans_th, ans_r, ans_c, ans_w, nbid

    def get_fnames(self):
        return self.day_idxes

    def get_day_fname(self, dayno):
        return self.day_idxes[dayno]

    def get_rev_scaler(self, day_no):
        """
        day_no: the current day idx
        """
        # return self.rev_scaler
        return 48./self.get_day_oracle(day_no)
        # return 48./self.oracle_revs[day_no]

    def randomize_days(self):
        self.day_idxes = np.random.permutation(self.day_idxes)
        print('randomized days {}'.format(self.day_idxes))

    def get_wkday_base(self):
        return 0

    def how_many_days(self):
        return self.num_days

# prepare synthetic data
def pre_run_slot_generator(cfg, mode, do_plot=False): # only consider syn
    """

    """
    data_seeds = eval(cfg['data']['data_seeds'])
    slot_len_in_min = eval(cfg['data']['slot_len'])
    train_days = eval(cfg['data']['train_days'])
    test_days = eval(cfg['data']['test_days'])
    num_sample_per_day = eval(cfg['data']['num_sample_per_day'])
    data_ver = cfg['data']['version']
    diverse_constraints = 'L' in data_ver
    iid = eval(cfg['data']['iid'])
    C = eval(cfg['data']['C'])
    budget = eval(cfg['data']['budget'])

    if budget>0 or diverse_constraints: # syn_v3 affect syn_v3b affect syn_v3bL
        savename = '{}_gen_meta_s{}_n{}{}_{}.pkl'.format(data_ver[:-1], slot_len_in_min,
                                                       num_sample_per_day,
                                                    '_b{}'.format(budget) if diverse_constraints else '', #bL需要基于b
                                                       mode)
        prior_meta = pkl.load(open('../data/' + savename, 'rb'))
    else: prior_meta = None
    budget_suffix = ''
    budgets_based_on_so_cost = None
    C_based_on_so_roi = None
    tmp_Ls = None
    if budget>0:
        if diverse_constraints: # meaning that budget has been specified before
            assert 'budgets' in prior_meta['day_stats']
            budgets_based_on_so_cost = prior_meta['day_stats']['budgets']
        else:
            budgets_based_on_so_cost = [round(x*budget) for x in prior_meta['day_stats']['slot_oracle']['cost']]
        budget_suffix = '_b{}'.format(budget)
    if diverse_constraints:
        reference = 'slot'
        C_based_on_so_roi = [np.floor(r/c*100)/100 for r,c in zip(prior_meta['day_stats']['{}_oracle'.format(reference)]['rev'],
                                                               prior_meta['day_stats']['{}_oracle'.format(reference)]['cost'])]
        tmp_Ls = [r/c for r, c in
         zip(prior_meta['day_stats']['{}_oracle'.format(reference)]['rev'],
             prior_meta['day_stats']['{}_oracle'.format(reference)]['cost'])]

    if mode=='train':
        data_seeds = data_seeds[:train_days]
        days = train_days
    else:
        assert train_days+test_days<=len(data_seeds)
        data_seeds = data_seeds[train_days:train_days+test_days]
        days = test_days
    generator = SlotGenerator(slot_len_in_min, train_days if mode == 'train' else test_days,
                              num_sample_per_day, iid, data_seeds, data_ver=data_ver, Ls=C_based_on_so_roi if diverse_constraints else C, budgets=budgets_based_on_so_cost)

    num_slot_in_day = (60 // slot_len_in_min) * 24
    slot_oracles, rev_by_slot, cost_by_slot, win_by_slot, nbid_by_slot = [[] for _ in range(5)]
    day_stats = dict(slot_oracle=dict(rev=[],cost=[],nwin=[]), nbid=[],
                     value_oracle=dict(rev=[],cost=[],nwin=[],ratio=[]),
                     rev_oracle=dict(rev=[], cost=[], nwin=[], ratio=[]),
                     avg_value=[],
                     dynamics=[],
                     corresponding_yewu=generator.sampler.selected_days,
                     budgets=[],
                     Ls=[]
                     )
    threshes = np.arange(0.3, 2.3, 0.01)
    # if do_plot:
    #     target_day = 20
    #     ndays = 10
    #     plot_keys = ['costprice_mean', 'rev_mean', 'avg_ctr']
    #
    #     fig, axes = plt.subplots(len(plot_keys)+1, ndays)
    #     fig.set_size_inches((10 * ndays, 6*(len(plot_keys)+1)))
    #     limits = [(np.inf, 0)] * (len(plot_keys)+1)

    for day in tqdm(range(days)):
        print(generator.sampler.selected_days[generator.get_day_fname(day)])
    #     ###################
    #     if do_plot:
    #         keys = ['rev','costprice']
    #         day_plot = {key+suffix:[] for key in keys for suffix in ['_mean','_mode']}
    #         day_plot['avg_ctr'] = []
            # day_plot['ctr_mean'] = []
            # if day==target_day+1:
            #     fig,axes = plt.subplots(3,4)
            #     fig.set_size_inches((4*5, 3*5))
            #     for idx in range(12):
            #         g = imread('g{}.png'.format(idx))
            #         ridx = idx//4
            #         cidx = idx%4
            #         axes[ridx,cidx].imshow(g)
            #     [ax.set_axis_off() for ax in axes.ravel()]
            #     plt.tight_layout()

                # plt.savefig('syn_cost_rev_joint2.pdf', bbox_inches='tight')
                # exit()
        ##################

        for slot in range(num_slot_in_day):
            _, df = generator.get_day_slot_data(day, slot, drop=slot-1, threshes=threshes, do_plot=do_plot)
            ######################
            # if do_plot:
            #     for key in keys:
            #         tmp = df[key]
            #         tmp = tmp[tmp>0]
            #         day_plot[key+'_mean'].append(tmp.mean())
            #         day_plot[key + '_mode'].append(tmp.mode())
            #     day_plot['avg_ctr'].append(df.click.sum()/df.shape[0])
            ######################

        ########################
        # if do_plot:
        #     if day<ndays:
        #         colidx = day
        #         axes[0, colidx].set_title('{}-{}'.format(day, generator.sampler.selected_days[day].split(os.path.sep)[-1]))
        #         for rowidx, key in enumerate(plot_keys):
        #             tmp = day_plot[key]
        #             ax = axes[rowidx, colidx]
        #             ax.set_ylabel(key)
        #             ax.plot(tmp) # don't plot modes, there are multi modes
        #
        #             lo, hi = limits[rowidx]
        #             loo, hii = np.min(tmp), np.max(tmp)
        #
        #             if loo < lo: lo = loo
        #             if hii > hi: hi = hii
        #             limits[rowidx] = (lo, hi)
        #             if rowidx == 1:
        #                 y = tmp
        #                 ysmoothed = gaussian_filter1d(y, sigma=1)
        #                 ax = axes[3, colidx]
        #                 ax.set_ylabel(key + '_smooth')
        #                 ax.plot(ysmoothed)
        #                 limits[3] = limits[1]

            # if day==ndays-1:
            #     [axes[idx, i].set_ylim(lo * 0.95, hi * 1.05) for idx, (lo, hi) in enumerate(limits) for i in
            #      range(ndays)]
            #     [axes[i, j].grid(True) for i in range(len(axes)) for j in range(ndays)]
            #
            #     plt.savefig('syn_cost_rev_ctr.pdf', bbox_inches='tight')
            #     exit()

        #######################

        self = generator
        day_stats['dynamics'].append(generator.dynamics)
        revsum = sum(self.slotoracle_rev_by_slot.values())
        costsum = sum(self.slotoracle_cost_by_slot.values())
        winsum = sum(self.win_by_slot.values())
        bidsum = sum(self.nbid_by_slot.values())
        vsum = sum(self.slot_value_sum.values())
        avg_value = vsum / bidsum
        day_stats['avg_value'].append(avg_value)
        day_stats['slot_oracle']['rev'].append(revsum)
        day_stats['slot_oracle']['cost'].append(costsum)
        day_stats['slot_oracle']['nwin'].append(winsum)
        day_stats['nbid'].append(bidsum)
        k2_rev = revsum
        k2_cost = costsum
        k2_win = winsum
        slot_oracles.append(generator.slot_oracles)
        rev_by_slot.append(generator.rev_by_slot)
        cost_by_slot.append(generator.cost_by_slot)
        win_by_slot.append(generator.win_by_slot)
        nbid_by_slot.append(generator.nbid_by_slot)
        oracle_revcost_by_slot_thrsh = generator.oracle_revcost_by_slot_thrsh
        # value oracle and rev oracle
        for k in ['rev_oracle']:
            if k=='rev_oracle':
                slot_thrsh_rcw = np.stack([x[:,:3] for x in oracle_revcost_by_slot_thrsh], axis=1)
            else:
                slot_thrsh_rcw = np.stack([np.stack([x[:,-1],x[:,1],x[:,2]], axis=-1) for x in oracle_revcost_by_slot_thrsh], axis=1)
            # slot_thrsh_rcw = np.stack(oracle_revcost_by_slot_thrsh, axis=1)  # thrsh,slot, 3
            rcw_thrsh = slot_thrsh_rcw.sum(1)  # thrsh, 3
            oracle_rev_thr = rcw_thrsh[:,0]
            roi_thrsh = rcw_thrsh[:, 0] / (rcw_thrsh[:, 1] + 1e-4)
            indices = np.arange(len(roi_thrsh))
            if diverse_constraints:
                C = C_based_on_so_roi[generator.get_day_fname(day)]
                day_stats['Ls'].append(C)
            print('current L {} (vs {})'.format(C, tmp_Ls[day] if tmp_Ls is not None else None))
            bool_arr = roi_thrsh > C
            if budgets_based_on_so_cost is not None:
                current_budget = budgets_based_on_so_cost[generator.get_day_fname(day)]
                day_stats['budgets'].append(current_budget)
                bool_arr = np.logical_and(rcw_thrsh[:, 1] <= current_budget , bool_arr)
                print('current budget {}'.format(current_budget))
            if bool_arr.sum()!=0:
                min_idx = oracle_rev_thr[bool_arr].argmax()
                ans_idx = indices[bool_arr][min_idx]
                ans_thresh = threshes[bool_arr][min_idx]
                rev,cost, win = rcw_thrsh[ans_idx]
            else: rev,cost,win,ans_thresh = 0,0,0,0
            day_stats[k]['rev'].append(rev)
            day_stats[k]['cost'].append(cost)
            day_stats[k]['nwin'].append(win)
            day_stats[k]['ratio'].append(ans_thresh)
            oracle_rev = rev
            gap = k2_rev-oracle_rev
            print('{} thresh {}, roi:{}, wr:{}, bidratio {}, oraclerev {}'.format(k, ans_thresh, rev/cost, win/bidsum,  ans_thresh, rev))
            print('slot oracle roi:{}, wr:{}, rev:{}, gap: {}({}%)'.format(k2_rev/k2_cost, k2_win/bidsum, k2_rev, gap, round(gap/oracle_rev*100,2),))
            # tmp = rcw_thrsh[:,1][bool_arr]
            # print(tmp.min(), tmp.max())
            # oracle_cost_each_slot = slot_thrsh_rcw[ans_idx, :, 1]
            # oracle_consumption_by_slot = np.cumsum(oracle_cost_each_slot)/self.budget
            # plt.figure()
            # plt.grid(True)
            # plt.plot(oracle_consumption_by_slot)
            # plt.show()

    day_slot = dict(slot_ratios=slot_oracles, rev=rev_by_slot, cost=cost_by_slot, win=win_by_slot, nbid=nbid_by_slot)

    savename = '{}_gen_meta_s{}_n{}{}_{}.pkl'.format(data_ver, slot_len_in_min, num_sample_per_day, budget_suffix, mode)
    savepath = os.path.join('../data', savename)
    pkl.dump(dict(day_stats=day_stats, day_slot_meta=day_slot), open(savepath, 'wb'))
    print('dump {}'.format(savepath))


class Simulator(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, mode='train', cfg=None, data_folder=None):
        """

        """
        super().__init__()
        self._load_config(cfg)
        budget = self.budget
        if budget <= 0:
            budget_suffix = ''
        else:
            budget_suffix = '_b{}'.format(budget)
        self.istest = mode=='test'
        if self.loading == 'gen':
            assert self.train_days<len(self.data_seeds), 'length of data seeds {} should be larger than train days {}'.format(len(self.data_seeds), self.train_days)
            if not self.istest :
                self.data_seeds = self.data_seeds[:self.train_days]
            else:
                self.data_seeds = self.data_seeds[self.train_days:]
        self._ptr = 0
        self.slot_no = 0
        self.budgets = None
        data_ver = cfg['data']['version']
        self.has_budget, self.has_roi_constraint = 'b' in data_ver, 'L' in data_ver or 'f' in data_ver
        setting = data_ver.split('_')[-1][2:].strip('ood') # eg. yewu_v4bL

        if self.loading=='each':
            train_files = eval(self.cfg['data'].get('{}_files'.format(mode), 'None'))
            self.reader = FileReader(data_folder, self.slot_len_in_min, self.cat_labels, mode, self.train_days, train_files,
                                     drop=eval(self.cfg['data']['drop']), setting=setting, do_plot=self.do_plot)
            self.budgets = self.reader.budgets
            self.Ls = self.reader.Ls
        else: # slot
            load_path = os.path.join('../data', '{}_gen_meta_s{}_n{}{}_{}.pkl'.format(data_ver, self.slot_len_in_min, self.num_sample_per_day, budget_suffix, mode))
            self.gen_meta = None
            budgets = None
            if os.path.exists(load_path):
                self.gen_meta = pkl.load(open(load_path,'rb'))

            if self.has_budget: # diverse budgets
                budgets = self.gen_meta['day_stats'].get('budgets', None)
                self.budgets = [0 for _ in range(100)] if budgets == [] else budgets
            else: self.budgets = [0 for _ in range(100)]
            if self.has_roi_constraint: # diverse constraints
                if 'Ls' in self.gen_meta['day_stats']:
                    self.Ls = self.gen_meta['day_stats']['Ls']
                else: self.Ls = [1. for _ in range(100)] # always 1
            else: self.Ls = [0 for _ in range(100)]
            data_seeds = self.data_seeds[:self.train_days] if mode=='train' else self.data_seeds[self.train_days:self.train_days+self.test_days]
            self.reader = SlotGenerator(self.slot_len_in_min, self.train_days if mode == 'train' else self.test_days,
                                        self.num_sample_per_day, self.iid, data_seeds, gen_meta=self.gen_meta,
                                        data_ver=data_ver, Ls=self.Ls, budgets=budgets)

        self.oracle_ratios, self.cost_by_slot, self.max_oracle_ratio, self.min_oracle_ratio, self.all_bid_requests, self.cat_sizes = self.reader.get_meta()

        self.reference_keys.append('agent')
        # if self.use_syn(): self.reference_keys.append('agent_fake')
        if self.not_full_data(): self.reference_keys.append('agent_po')
        wkday_base = self.reader.get_wkday_base()
        self.day2wkday = wkday_base
        self.wkday2day = -wkday_base
        # self.max_day = self.bid_requests.day_no.max()
        self.day_ckpt = 0
        self.mode = mode

        self.total_slots = self.reader.how_many_days() * 24 * 60 // self.slot_len_in_min
        self.bid_line = {}

        self.reset(reset_curday=False)

        self.slot2revcosts = {}

        self.state_preserver = StateController(self.num_slot_in_day, self.cfg,
                                               budgets=self.budgets,
                                               constraints=self.Ls
                                               )
        self.reset_traversal()
        self.sar_size = self.state_preserver.get_obs_size()+2 if self.history_type=='sar' else self.state_preserver.get_obs_size()*2+2

        self.reward_type = eval(cfg['rl_agent']['rewardtype'])
        self.penalty_scale = self.init_penalty_scale = float(cfg['hypers']['penalty'])  # float(cfg['agent']['penalty'])
        self.reward_scale = eval(cfg['data']['rev_scaler'])  # 1e4/20 500
        self.init_lambda = eval(
            self.cfg['rl_agent']['init_lambda'])  # 1.6 if 'yewu' not in self.envname.lower() else 1.6
        pid_ratio = eval(self.cfg['pid']['ratio'])
        if pid_ratio!=0:
            self.init_lambda = pid_ratio
            print('using initial ratio {}'.format(pid_ratio))

        self.agent_ratios = dict()
        self.L_margin = 0.2
        self.b_margin = 0.05 # 允许提前花多少budget
        self.indicator_approx_v = 10.
        self.sqrt_v = np.sqrt(self.indicator_approx_v)

    def get_record_length(self):
        return len(self.reference_all_revs['agent'])

    def save(self, output_dir):
        pkl.dump(self.agent_ratios, open(os.path.join(output_dir, 'agent_ratios_{}.pkl').format('test' if self.istest else 'train'), 'wb'))

    def set_gamma(self, gamma):
        self.relax_gamma = gamma
        self.logger.info('setting relax gamma {}'.format(gamma))

    def set_b(self, gamma):
        self.L_margin = gamma
        self.logger.info('setting L margin {}'.format(gamma))

    def get_day_oracle(self):
        dayno = (self.slot_no-1) // self.num_slot_in_day
        return self.reader.get_day_oracle(dayno)

    def get_obs(self, ratio, empty, force_no_history=False, avg_value=False, iter=None):
        """
        return:
            if use_history: (current obs, history, length)
            else: current obs
        """
        dayno = (self.slot_no) // self.num_slot_in_day
        if empty:
            size = self.state_preserver.get_obs_size()
            pad = np.zeros(size if self.position_encoding==0 else size+1, dtype=np.float32)
            if self.use_history and not force_no_history:
                return (pad, np.zeros(self.sar_size, dtype=np.float32), 0)
            else: return pad

        revcost = self.get_revcost()
        if avg_value:
            df = self.reader.get_day_data(dayno)
            avg_value = (df.pppc * df.pctr ).sum()
        else: avg_value = None
        # df = self.reader.get_day_data(dayno) # envoke reading
        obs = self.state_preserver.get_observation(dict(
            avg_value=avg_value,
            iter=iter,
            dayno=self.reader.get_day_fname(dayno),
            ratio=ratio,
            revcost=revcost,
            scaler=self.reader.get_rev_scaler(dayno), # 1/(total_rev/numslot) ==> divided by average slot-wise oracle rev

        ))
        if self.use_history and not force_no_history:
            history = self.state_preserver.get_history()
            if history: # not None, then list of tuples
                history,num_history = history
                # history: list of nparrays
                history = np.stack(history)
                return (obs,history,num_history)
            else: return (obs, np.zeros((1,self.sar_size), dtype=np.float32), 0)
        else: return obs

    def get_obs_cem(self, empty):
        if empty:
            size = self.state_preserver.get_obs_size()
            return np.zeros(size)
        return self.get_observation_cem()

    def get_revcost(self, num_future=2):# todo: 如果是synthetic cost，下面的特征实际上不准的
        """
        涉及当前slot，需要超前读一天
        """
        this_dayno, this_slotno = self.get_slotno_parts(slotid=self.slot_no)
        last_dayno, last_slotno = self.get_slotno_parts()
        df = self.reader.get_day_data(this_dayno, drop=last_dayno-1)
        if self.loading=='slot':
            self.reader.get_day_slot_data(this_dayno, this_slotno, drop=this_slotno-(self.slot_len_in_min+num_future)) # slot_len + num_future -> drop 0
        if last_dayno != this_dayno:
            last_rev, last_cost = 0, 0
        else:
            last_rev, last_cost = self.reader.get_rev_by_dayslot(last_dayno,
                                                                 last_slotno), self.reader.get_cost_by_dayslot(
                last_dayno, last_slotno)
        this_rev, this_cost = self.reader.get_rev_by_dayslot(this_dayno, this_slotno), self.reader.get_cost_by_dayslot(
            this_dayno, this_slotno)
        ret = [last_rev, last_cost, this_rev, this_cost]
        if num_future>0:
            for idx in range(1, num_future+1):
                n1_dayno, n1_slotno = self.get_slotno_parts(slotid=self.slot_no+idx)
                if n1_dayno != this_dayno:
                    n1_rev, n1_cost = 0, 0
                else:
                    n1_rev = self.reader.get_rev_by_dayslot(n1_dayno, n1_slotno)
                    n1_cost = self.reader.get_cost_by_dayslot(n1_dayno, n1_slotno)
                ret.extend([n1_rev, n1_cost])
        return ret

    def get_slot_oracle_ratio(self, dayno, slot_no):
        return self.reader.get_slot_oracle_ratio(dayno, slot_no)

    def randomize_days(self):
        # self.bid_requests_list = np.random.permutation(self.bid_requests_list)
        self.reader.randomize_days()
        self.logger.info('randomized days {}'.format(self.reader.get_fnames()))

        # return tuple()

    def _load_config(self, cfg):
        """
        Parse the config.cfg file
        """
        # cfg = configparser.ConfigParser(allow_no_value=True)
        # env_dir = os.path.dirname(__file__)
        # cfg.read('./config.cfg')
        self.cfg = cfg
        self.data_src = cfg['data']['dtype']
        self.include_loosing = eval(cfg['data']['include_loosing'])
        self.use_synthetic_cost = cfg['data']['use_syn']
        # self.file_in = '../data/yewu-data' #os.getcwd()+'/'+cfg['data']['yewu_path']
        self.metric = str(cfg['data']['metric'])
        # self.mode = cfg['data']['mode']

        self.slot_len_in_min = eval(cfg['data']['slot_len'])
        self.num_slot_in_period = eval(cfg['data']['period_len'])
        self.num_slot_in_hour = 60 // self.slot_len_in_min if self.slot_len_in_min<60 else 60/self.slot_len_in_min
        self.num_slot_in_day = 24*self.num_slot_in_hour if self.slot_len_in_min<60 else 24*60 // (self.slot_len_in_min)

        self.train_max_days = eval(cfg['data']['train_max_days'])
        self.envname = cfg['data']['envname']

        # self.istest = 'test' in self.envname
        self.cem_nsample = eval(cfg['cem']['exp_nsample'])
        self.cem_exp_percent = eval(cfg['cem']['exp_percent'])
        self.gamma = eval(cfg['rl_agent']['gamma'])
        self.loading = cfg['data']['loading']
        self.cat_labels = eval(self.cfg['rl_agent']['cat_labels'])  # 'pid'
        self.debug = eval(self.cfg['data']['debug'])
        self.train_niter = eval(self.cfg['data']['train_niter'])
        self.data_folder = self.cfg['data']['data_folder']
        self.train_days = eval(self.cfg['data']['train_days'])
        self.test_days = eval(self.cfg['data']['test_days'])
        self.iid = eval(self.cfg['data']['iid'])
        self.num_sample_per_day = eval(self.cfg['data']['num_sample_per_day'])
        self.exp_seed = eval(self.cfg['data']['seed'])
        self.data_seeds = eval(self.cfg['data']['data_seeds'])
        self.num_bids_each_bin = eval(self.cfg['cem']['num_bids_each_bin'])
        self.position_encoding = eval(self.cfg['rl_agent']['position_encoding'])
        self.norm_reward = eval(self.cfg['rl_agent']['norm_reward'])
        self.use_history = eval(self.cfg['rl_agent']['use_history'])
        self.history_type = self.cfg['rl_agent']['history_type']
        # self.Ls = eval(self.cfg['data']['C'])
        self.data_ver = self.cfg['data']['version']
        self.budget = eval(cfg['data']['budget'])
        self.budget_braking = eval(cfg['rl_agent']['budget_braking'])
        # if 'yewu' in self.data_ver: self.reference_keys.append('k2')
        self.is_syn = 'syn' in self.data_ver
        self.reference_keys = ['slot_oracle','k2'] if not self.is_syn else ['oracle','slot_oracle']
        self.do_plot = eval(cfg['data']['draw'])
        # self.num_day = eval(cfg['data']['day_num'])
    def set_reward_scaler(self, scaler):
        self.state_preserver.set_reward_scaler(scaler)

    def get_slot_start(self):
        max_day = self.how_many_days() - 1
        if max_day+1>self.train_max_days:
            min_day = max_day+1-self.train_max_days
        else: min_day = 0

        return min_day,0

    def get_day_rsum(self, dayno):
        return self.reader.get_day_rsum(dayno)

    def not_full_data(self):
        return (not self.include_loosing or self.use_syn()) and not self.istest

    def use_syn(self):
        return self.use_synthetic_cost!='none' and self.include_loosing

    # trick: step>1 so that past_x_ends() will not be enabled at the beginning
    def reach(self, ptr):
        return self.slot_no == ptr

    def get_past_slot_day(self):
        return (self.slot_no - 1) // self.num_slot_in_day

    def get_current_slot(self):
        return (self.slot_no-1) % self.num_slot_in_day

    def test_period_ends(self):
        current_slot = self.slot_no
        return (current_slot)%self.num_slot_in_period==0

    def past_day_ends(self):
        # if self.new_day_start and self._ptr>=int(self.total_bids / 7. * 0.9): return True
        # else: return False
        return (self.slot_no)%self.num_slot_in_day==0

    def past_hour_ends(self):
        # if self.new_hour_start and self._ptr>0: return True
        # else: return False
        return (self.slot_no)%self.num_slot_in_hour==0

    def past_week_ends(self):
        # if self.past_day_ends() and self.cur_day %7==0: return True
        # else: return False
        return (self.slot_no)%(self.num_slot_in_day*7)==0


    def get_histogram(self):
        self.hist_params = dict()
        # import ipdb; ipdb.set_trace()
        st = time.time()
        # for slot, group in self.bid_requests.groupby('time_slot'):
            # loosed_cost = group.costprice[group.loose]
        group = self.all_bid_requests
        won_cost = group.costprice[~group.loose.astype(bool)]
        self.hist_params = np.histogram(won_cost, bins=int(won_cost.max())-int(won_cost.min()), density=True)

        print('fit histogram cost model takes {}'.format(time.time()-st))

    def fit_mprice_model(self):
        """
        hist synthetic cost
        """
        if self.past_day_ends():
            if self.use_synthetic_cost=='hist' and not self.istest:
                dayno, slotno = self.get_slotno_parts(past=False)
                df = self.reader.get_day_data(dayno)
                won_data = df.costprice[~df.loose].values
                self.mprice_model.update_hist(won_data)

    def save_mprice_model(self):
        if self.use_synthetic_cost=='hist' and not self.istest:
            self.mprice_model.save()

    def set_mprice_model(self, model):
        self.mprice_model = model
        # print('censored model added')

    def make_cost_prediction(self): # todo: syn for which day?
        # if not self.use_loose(): return
        if self.use_synthetic_cost=='hist':
            st = time.time()
            for dtmp in self.bid_requests_list: # todo: correct for self.reader
            # self.bid_requests_list.loc[:, 'syn_cost'] = 600
            # for slot, group in self.bid_requests.groupby('time_slot'):
            #     densities, values = self.hist_params[slot]
                densities, values = self.hist_params
                # group = self.bid_requests_list
                tmp = np.random.choice(values[:-1], size=len(dtmp.values), replace=True, p=densities)
                tmp[tmp<600] = 600
                dtmp.loc[:, 'syn_cost'] = tmp

            print('sample from non-parametric cost model takes {}'.format(time.time()-st))
        elif self.use_synthetic_cost=='censored':
            for dtmp in self.bid_requests_list:
            # tmp = self.bid_requests_list
                dtmp.loc[:,'syn_cost'] = dtmp.costprice
                selector = dtmp.loose
                data = dtmp[selector]
                res = self.mprice_model.do_predict(data)
                dtmp.loc[selector, 'syn_cost'] = res

    def get_simulations(self, action_space, init_lambda, dayno):
        print('make simulations..')
        # simulations
        tmp = self.reader.get_day_data(dayno) #self.bid_requests_list[dayno]
        selector = np.logical_and(tmp.time_slot>=13*self.num_slot_in_hour, tmp.time_slot<self.num_slot_in_day)
        tmp = tmp[selector]

        # action_space = self._get_action_space()
        num_slots = len(tmp.time_slot.unique())
        num_sim = 500
        sim_actions = init_lambda * (
                1 + np.random.choice(action_space, size=(num_sim, num_slots), replace=True))
        all_slots = np.array(tmp.time_slot.unique())

        res = dict()

        st = time.time()
        # slot_list = list(tmp.time_slot.unique())
        # for slotid in slot_list:
            # if slotid not in res
        if len(all_slots)==0: return
        slotid = all_slots[0]
        if slotid in self.slot2revcosts: return
        min_slot = slotid
        # max_slot = (slotid//self.num_slot_in_day+1)*self.num_slot_in_day-1
        # rev_cumsum, cost_cumsum = foo(slotid)
        max_slot = (min_slot // self.num_slot_in_day + 1) * self.num_slot_in_day - 1
        # sel = np.logical_and(all_slots >= min_slot, all_slots <= max_slot)
        # slots = all_slots[sel]
        slots = all_slots
        # df = tmp[np.logical_and(tmp.time_slot >= min_slot, tmp.time_slot <= max_slot)]
        df = tmp
        actions = sim_actions#[:, sel]
        mapping = dict(zip(slots, map(pd.Series, actions.T)))
        ratios = df.time_slot.apply(lambda x: mapping[x])
        revs_arr = df.rev.values.reshape(-1, 1)
        costs_arr = df.costprice.values.reshape(-1, 1)
        value_arr = df.value.values.reshape(-1, 1)
        wins = ratios.values * value_arr > costs_arr

        wins_arr = wins.astype(int)  # num, numsim

        revsum = (wins_arr * revs_arr).cumsum(axis=0)  # numsim
        costsum = (costs_arr * wins_arr).cumsum(axis=0) * 1e-3
        indexes = np.arange(len(df))
        for sid in range(min_slot, max_slot+1):
            # df = tmp[np.logical_and(tmp.time_slot >= slotid, tmp.time_slot <= max_slot)]
            idxs = indexes[df.time_slot==sid]
            if len(idxs)>0:
                res[(dayno,sid)] = revsum[idxs[-1]],costsum[idxs[-1]]
        del df, mapping, ratios, revs_arr, costs_arr, value_arr, wins, wins_arr


        for dayno in range(1, self.how_many_days() + 1):
            res[(dayno,0)] = (np.zeros(1),np.zeros(1))
            res[(dayno-1, self.num_slot_in_day)] = (np.zeros(1), np.zeros(1))

        self.slot2revcosts.update(res)
        print('simulation takes', time.time()-st)

    def transfer_from_to(self, env, start, end):
        tmp = [self.reader.get_day_data(start)]#self.bid_requests_list[start:end]
        # data = tmp[np.logical_and(tmp.day_no>=start, tmp.day_no<end)]
        # tmp = data.copy()
        min_slot,max_slot = start * self.num_slot_in_day, end * self.num_slot_in_day
        env.oracle_rois = np.concatenate(
            [env.oracle_rois,
             self.oracle_rois[min_slot:max_slot]])
        env.cost_by_slot = np.concatenate(
            [env.cost_by_slot,
             self.cost_by_slot[min_slot:max_slot]]
        )
        env.oracle_ratios = np.concatenate(
            [env.oracle_ratios,
             self.oracle_ratios[start:end]]
        )
        # mapping = {v:idx+env.get_max_day()+1 for idx,v in enumerate(tmp.day_no.unique())}
        # tmp['day_no'] = tmp.day_no.map(mapping)
        slot_base = env.how_many_days()
        this_base = start
        offset = (slot_base - this_base)
        # involved = set(tmp.time_slot.unique())

        # tmp['time_slot'] = tmp.time_slot.apply(lambda x: x+offset) #
        # env.bid_requests = pd.concat([env.bid_requests, tmp], axis=0)
        env.bid_requests_list.extend(tmp)

        env.slot2revcosts.update({(dn+offset,sn):r for (dn,sn),r in self.slot2revcosts.items() if start<=dn<end})
        # loose according to policy ratio
        for idx in range(len(tmp)):
            dtmp = tmp[idx]
            bids = dtmp.agent * dtmp.value
            wins = bids>dtmp.costprice
            dtmp.loc[~wins,'loose'] = True
            dtmp.loc[wins, 'loose'] = False
        print('data ranging from {} to {} has been transferred'.format(start, end))
        # env.max_day += end - start

    def reset_traversal(self):
        # if self.include_loosing:
        #     ref_keys = self.reference_keys
        # else:
        #     ref_keys = self.reference_keys + ['agent_po']
        ref_keys = ['k2','oracle','slot_oracle', 'agent', 'agent_po']
        self.reference_all_revs = {k: [] for k in ref_keys}
        self.reference_all_revs['agent_normalized'] = []
        self.reference_all_revs['agent_po_normalized'] = []
        self.reference_all_costs = {k: [] for k in ref_keys}
        self.reference_all_wins = {k: [] for k in ref_keys}
        self.reference_all_rewards = {k: [] for k in ref_keys}
        self.reference_all_roi_err = {k: [] for k in ref_keys}
        # self.reference_all_rewards_fake = {k: [] for k in ref_keys}
        self.all_bids = {k: [] for k in ref_keys}
        self.all_constraint = []
        self.all_oracle = []
        self.day_valuesum = []


    def how_many_days(self):
        return self.reader.how_many_days()

    def get_wkday(self):
        day = self.get_day()
        return (self.day2wkday+day)%7

    def reset(self, reset_curday=True):
        """ initial state
        return: observation, last_action_reward, last_action_cost
        """
        if reset_curday:
            curday, self.curday_start_slot = self.get_slot_start()
            self.slot_no = curday * self.num_slot_in_day
        else:
            curday, self.curday_start_slot = 0, 0
            self.slot_no = 0
        self.new_day_start = False
        self.past_slot_wkday = -1

        self.cur_day = curday
        self.past_slot_day = -1
        self.new_hour_start = False
        self.cur_hour = 0
        self.past_hour = -1

        self._ptr = 0
        # self.slot_no = 0
        self.ratios = []

        # if self.use_syn(): self.sample_syn_cost()

    def update_timestamp(self, bid_hr, bid_wkday, bid_dayno):
        if self.past_hour_ends():
            self.new_hour_start = True


        if self.past_day_ends(): # depends on slot_no only
            self.new_day_start = True
            self.cur_day = self.get_day()

        self.past_slot_day = (self.slot_no - 1) // self.num_slot_in_day  # self.cur_day
        self.past_hour = ((self.slot_no-1) // self.num_slot_in_hour)% 24
        self.cur_hour = ((self.slot_no) // self.num_slot_in_hour) % 24

    def set_logger(self, logger):
        self.logger = logger
        self.reader.set_logger(logger)
        self.logger.info('{} dataset includes {}'.format(self.mode, self.reader.get_fnames()))

    def update_each_period(self, update_ratio=True):
        if self.loading=='slot': return
        dayno = (self.slot_no - 1) // self.num_slot_in_day
        ddf = self.reader.get_day_data(dayno)

        start_slot = self.curday_start_slot % self.num_slot_in_day
        end_slot = (self.slot_no-1) % self.num_slot_in_day + 1
        self.curday_start_slot = end_slot
        # self.ratios = []

    def dayover_stats_update(self):
        # if self.past_day_ends(): #and self.cur_day>len(self.all_costs): # new day start 因为具有延迟（past day ends，但是没有won pv)
        past_dayno = self.get_day() - 1

        if self.loading=='slot': # todo: slot loading, change name to slot_oracle
            past_dayno = self.reader.get_day_fname(past_dayno)
            # k2:
            key = 'slot_oracle'
            if self.gen_meta is None:
                rsum, csum, wsum, bsum = self.reader.stats_of_day()
            else:
                day_stats = self.gen_meta['day_stats']
                rsum = day_stats[key]['rev'][past_dayno]
                csum = day_stats[key]['cost'][past_dayno]
                wsum = day_stats[key]['nwin'][past_dayno]
                bsum = day_stats['nbid'][past_dayno]

            self.reference_all_costs[key].append(csum)
            self.reference_all_revs[key].append(rsum)
            self.reference_all_wins[key].append(wsum)
            self.all_bids[key].append(bsum)
            # oracle:
            if self.gen_meta is None:
                slot_thrsh_rcw = np.stack(self.reader.oracle_revcost_by_slot_thrsh, axis=1) # thrsh,slot, 3
                rcw_thrsh = slot_thrsh_rcw.sum(1) # thrsh, 3
                roi_thrsh = rcw_thrsh[:,0]/(rcw_thrsh[:,1]+1e-4)
                indices = np.arange(len(roi_thrsh))
                bool_arr = roi_thrsh>self.state_preserver.get_day_constraint(past_dayno)
                min_idx = roi_thrsh[bool_arr].argmin()
                ans_idx = indices[bool_arr][min_idx]

                rsum,csum,wsum = rcw_thrsh[ans_idx]
            else:
                if 'syn' in self.data_ver:
                    k = 'rev_oracle'
                else:
                    k = 'oracle'
                rsum = day_stats[k]['rev'][past_dayno]
                csum = day_stats[k]['cost'][past_dayno]
                wsum = day_stats[k]['nwin'][past_dayno]
                # bsum = day_stats['nbid'][past_dayno]
            self.reference_all_costs['oracle'].append(csum)
            self.reference_all_revs['oracle'].append(rsum)
            self.reference_all_wins['oracle'].append(wsum)
            self.all_bids['oracle'].append(bsum)
        else: # todo: k2, slot_oracle, oracle
            policies = self.reader.policies # policy 2 stats; nbid
            dayid = self.reader.get_day_fname(past_dayno)
            bidnum = policies[dayid]['nbid']
            for k in ['k2', 'oracle', 'slot_oracle']:
                revsum,costsum,winsum = [policies[dayid][k][x] for x in ['rev','cost','nwin']]
                self.reference_all_revs[k].append(revsum)
                self.reference_all_costs[k].append(costsum)
                self.reference_all_wins[k].append(winsum)
                self.all_bids[k].append(bidnum)

        if self.not_full_data():
            self.reference_all_revs['agent_po'].append(self.state_preserver.rev_e)
            self.reference_all_costs['agent_po'].append(self.state_preserver.cost_e)
            self.reference_all_wins['agent_po'].append(self.state_preserver.wins_e)
            self.all_bids['agent_po'].append(self.state_preserver.bids_e)
            self.reference_all_rewards['agent_po'].append(self.state_preserver.reward_e)
            self.reference_all_revs['agent_po_normalized'].append(min(self.reference_all_revs['agent_po'][-1] / self.reference_all_revs['oracle'][-1]*1e2,102))
            roi = self.state_preserver.rev_e/(1e-4+self.state_preserver.cost_e)
            roi_err = roi - self.get_curday_constraint()
            self.reference_all_roi_err['agent_po'].append(roi_err)


        k = 'agent'
        self.reference_all_revs[k].append(self.state_preserver.rev_e_rw)
        self.reference_all_costs[k].append(self.state_preserver.cost_e_rw)
        self.reference_all_wins[k].append(self.state_preserver.wins_e_rw)
        self.all_bids[k].append(self.state_preserver.bids_e_rw)
        self.reference_all_rewards[k].append(self.state_preserver.reward_e_rw)
        roi = self.state_preserver.rev_e_rw/(1e-4+self.state_preserver.cost_e_rw)
        constraint = self.get_curday_constraint()
        roi_err = roi - constraint
        self.reference_all_roi_err[k].append(roi_err)
        self.all_constraint.append(constraint)
        self.all_oracle.append(self.get_day_oracle())

        self.reference_all_revs['agent_normalized'].append(min(self.reference_all_revs['agent'][-1] / self.reference_all_revs['oracle'][-1] * 1e2,102))
        # if 'agent_fake' in self.reference_all_rewards:
        #     self.reference_all_rewards['agent_fake'].append(self.state_preserver.reward_e)
        self.exe_rev = self.state_preserver.exe_rev
        self.exe_cost = self.state_preserver.exe_cost
        self.agent_ratios[self.get_day_fname()] = self.ratios
        self.ratios = []

    def get_k2_info(self):
        # import ipdb; ipdb.set_trace()
        all_keys = ['k2','oracle','slot_oracle']
        ret = [self.get_day_fname()]
        if len(self.reference_all_revs[self.reference_keys[0]])>0:

            for k in self.reference_keys:
                if k in self.reference_all_revs:
                    ret.extend([k, self.reference_all_revs[k][-1], self.reference_all_costs[k][-1], self.reference_all_revs[k][-1] / (self.reference_all_costs[k][-1] + 1e-4)])
        else:
            for k in self.reference_keys:
                ret.extend([k, 0., 0., 0.])
        return ret

    def get_curday_constraint(self):
        """
        past day's constraint
        """
        dayno, slotno = self.get_slotno_parts()
        correct_dayno = self.reader.get_day_fname(dayno)
        C = self.state_preserver.get_day_constraint(correct_dayno)
        return C

    def get_day_budget(self, dayno):
        correct_dayno = self.reader.get_day_fname(dayno)
        C = self.state_preserver.get_day_budget(correct_dayno)
        return C

    def get_curday_budget(self):
        dayno, slotno = self.get_slotno_parts()
        return self.get_day_budget(dayno)

    def get_step(self): # todo: return multiple day same slot
        return self.slot_no

    def get_day(self):
        tmp = ((self.slot_no)//(self.num_slot_in_day))
        self.cur_day = tmp
        return tmp

    def get_day_nomod(self):
        return self.slot_no // self.num_slot_in_day

    def set_step(self, slot_no):
        self.slot_no = slot_no

    def compute_reward_cem(self, revsum, costsum, num_bids):
        dayno, slotno = self.get_slotno_parts()
        correct_dayno = self.reader.get_day_fname(dayno)
        current_rbr = self.remaining_budget_rate(correct_dayno)  # day_no is the correct one
        C = self.state_preserver.get_day_constraint(correct_dayno)
        roi = revsum/(costsum+1e-4)
        # if self.reward_type==5:
        if roi>=C:  #todo: diverse constraints for cem
            reward = revsum
            normalized_reward = revsum / num_bids
        else:
            reward = (roi - C)
            normalized_reward = reward
        return reward, normalized_reward

    def get_reward_sub_types(self):
        base = 9
        choice = self.reward_type - base
        dense_aux_accum = choice // 100
        aux = 1 if dense_aux_accum == 1 else 0
        accum = 1 if dense_aux_accum == 2 else 0
        combine_type = (choice % 100) // 10
        gap_type = choice % 10
        return combine_type, gap_type, aux, accum

    def compute_reward(self, day_no, slot_no, *args, real=False):
        if self.reward_type==0: # monte carlo simulation with soft combination
            return self.montecarlo_soft_reward(*args)
        elif self.reward_type==1: # monte carlo simulation with hard barrier
            return self.montecarlo_hard_reward(*args)
        elif self.reward_type==2: # deprecated
            return self.compute_reward_demo(day_no, slot_no, real)
        elif self.reward_type==3: # soft combination, sparse reward
            return self.compute_reward_sparse(day_no, slot_no, alpha=args[0], real=real, soft=True) # args0 is the alpha
        elif self.reward_type==4: # (part of) ICM reward
            return self.compute_reward_curiosity(day_no, slot_no, real=real)
        elif self.reward_type==5: # reward shaping
            return self.compute_reward_shaped2(day_no, slot_no, real)
        elif self.reward_type==6: # proposed: hard barrier, dense reward (curriculum learning)
            return self.compute_reward_powerlaw(day_no, slot_no, limit_param=args[1], real=real)
        elif self.reward_type==7: # hard barrier, sparse reward
            return self.compute_reward_sparse(day_no, slot_no, real=real, soft=False)
        elif self.reward_type==8:# deprecated
            return self.compute_reward_prototype_shaped2(day_no, slot_no, real)
        else: # deprecated
            combine_type, gap_type, aux, accum = self.get_reward_sub_types()
            return self.compute_reward_dense(real=real, combine_type=combine_type, gap_type=gap_type, aux=aux, accum=accum)

    def montecarlo_soft_reward(self, slot_rs, slot_cs, oracle_rev, upper=50, C=None, gamma=None):
        dayno, slotno = self.get_slotno_parts()
        correct_dayno = self.reader.get_day_fname(dayno)
        if C is None: C = self.state_preserver.get_day_constraint(correct_dayno)
        slot_iors = slot_cs / (slot_rs + 1e-4)
        # ROI>c => IOR<1/C => IOR/1/C < 1
        power = np.asarray([max(x,0) for x in np.clip(slot_iors, 0., upper) / (1./C) - 1.])
        if gamma is None: gamma = self.gamma
        penalties = gamma ** power - 1
        # target_v = min(episode_r/oracle_rev, 1.) - penalty
        target_vs = np.clip(slot_rs / oracle_rev, -np.inf, 1.) - penalties
        return target_vs

    def montecarlo_hard_reward(self, slot_rs, slot_cs, oracle_rev):
        dayno, slotno = self.get_slotno_parts()
        correct_dayno = self.reader.get_day_fname(dayno)
        C = self.state_preserver.get_day_constraint(correct_dayno)

        slot_rois = slot_rs / (slot_cs + 1e-4)
        target_vs = np.zeros_like(slot_rois)
        selector = slot_rois>=C
        nselector = slot_rois<C
        normalized_rev = np.clip(slot_rs/oracle_rev, 0., 1.)

        target_vs[selector] = normalized_rev[selector]
        target_vs[nselector] = (slot_rois - C)[nselector]

        if self.budget>0:
            budget = self.get_day_budget(dayno)
            budget_nselector = slot_cs > budget
            both_nselector = nselector & budget_nselector
            beyond_budget = (1-slot_cs/budget )
            target_vs[both_nselector] += beyond_budget[both_nselector]
            budget_only_nselector = selector & budget_nselector
            target_vs[budget_only_nselector] = beyond_budget[budget_only_nselector]

        return target_vs

    def compute_reward_shaped(self, day_no, slot_no, real=False): # action: 0不变，1下调，2上调
        if not real:
            rev, cost, last_roi = self.state_preserver.rev_e, self.state_preserver.cost_e, self.state_preserver.last_step_ROI_e
            roi_e = rev / cost
            rev_t = self.state_preserver.rev_t
            # roi_t = self.state_preserver.rev_t / self.state_preserver.cost_t
        else:
            rev, cost, last_roi = self.state_preserver.rev_e_rw, self.state_preserver.cost_e_rw, self.state_preserver.last_step_ROI_e_rw
            roi_e = rev / cost
            rev_t = self.state_preserver.rev_t_rw

        dayno, slotno = self.get_slotno_parts()
        correct_dayno = self.reader.get_day_fname(dayno)
        C = self.state_preserver.get_day_constraint(correct_dayno)

        slot_no = (self.slot_no - 1) % self.num_slot_in_day
        rew = 0
        if slot_no==self.num_slot_in_day-1:
            if roi_e >= C: rew += 100*np.log2(rev*1e-3)
            else: rew += -100

        action = self.last_action_type

        if last_roi < C and action==2: rew += -1
        if last_roi > C and action==1: rew += -1
        # if roi_e < self.C: rew += slot_no*(-10)
        return rew

    def compute_reward_demo(self, day_no, slot_no, real=False):
        if not real:
            rev, cost, last_roi = self.state_preserver.rev_e, self.state_preserver.cost_e, self.state_preserver.last_step_ROI_e
        else:
            rev, cost, last_roi = self.state_preserver.rev_e_rw, self.state_preserver.cost_e_rw, self.state_preserver.last_step_ROI_e_rw
        current_ROI_e = rev / (cost + 1e-4)
        env = self
        # inc = 0.
        slot_no = (self.slot_no-1) % self.num_slot_in_day
        oracle_roi = env.oracle_rois[env.slot_no-1]
        value = -np.abs(current_ROI_e-oracle_roi) * (slot_no * self.state_preserver.slot_scaler)
        # import ipdb; ipdb.set_trace()
        rnet_r = self.reward_scale * self.state_preserver.rev_t + self.penalty_scale * value
        # value: 0~1; rev_t: 0~1000
        self.state_preserver.penalty_meter.add(value)
        return rnet_r

    def core_reward_compute(self, current_ROI_e, rev_t, C, current_bcr=None):
        if current_ROI_e<C:
            gap = min(current_ROI_e, C) - C
            rev_t = 0
        else:
            gap = 0

        if current_bcr is not None and current_bcr<0:
            rev_t = 0
            gap += current_bcr # todo: currently linearly combine

        return gap, rev_t

    def get_recent_roi(self, real=False):
        if not real:
            rev, cost = self.state_preserver.rev_e, self.state_preserver.cost_e
            current_roi = rev / (cost+1e-4)
            last_roi = self.state_preserver.last_step_ROI_e
        else:
            rev, cost = self.state_preserver.rev_e_rw, self.state_preserver.cost_e_rw
            current_roi = rev / (cost + 1e-4)
            last_roi = self.state_preserver.last_step_ROI_e_rw
        return current_roi, last_roi

    def get_rev_scaler(self, dayno):
        """
        dayno: the current day idx, not the original idx
        """
        return self.reader.get_rev_scaler(dayno) # self.reward_scale

    def get_slotno_parts(self, past=True, slotid=None):
        if past: slot_no = self.slot_no-1
        else: slot_no = self.slot_no
        if slotid is not None: slot_no = slotid
        dayno = slot_no // self.num_slot_in_day
        slotno = slot_no % self.num_slot_in_day
        return dayno, slotno

    def compute_reward_prototype_shaped2(self, day_no, slot_no, real=False): # action: 0不变，1下调，2上调
        current_roi, last_roi = self.get_recent_roi(real)
        dayno,slotno = self.get_slotno_parts()
        correct_dayno = self.reader.get_day_fname(dayno)
        C = self.state_preserver.get_day_constraint(correct_dayno)
        rev_t = (self.state_preserver.rev_t if not real else self.state_preserver.rev_t_rw)
        gap, _ = self.core_reward_compute(current_roi, rev_t, C)  # gap<=0
        self.state_preserver.penalty_meter.add(gap)

        if slotno+1==self.num_slot_in_day: thresh = 0.
        else: #
            thresh = (1-(slotno+1)/self.num_slot_in_day)**1*(1-self.relax_gamma)* 0.5*C
        rev_indicator = abs(gap)<=thresh
        penalty_indicator = abs(gap)>thresh # 约往后越小
        gap = thresh-abs(gap)
        factor = self.gamma ** (self.num_slot_in_day - 1 - slotno)
        gap *= factor

        rev_scale = self.reward_scale # self.get_rev_scaler(dayno)
        rnet_r = rev_scale * rev_t * rev_indicator + self.penalty_scale * gap * penalty_indicator

        return rnet_r

    def remaining_budget_rate(self, day_no):
        budget = self.state_preserver.get_day_budget(day_no)  # only need the correct day_no
        if budget==0: return 0
        else:
            bcr = self.state_preserver.remaining_budget / budget
            return bcr

    def compute_reward_powerlaw(self, day_no, slot_no, limit_param=None, real=False): # action: 0不变，1下调，2上调
        # don't use penalty scaler
        current_roi, last_roi = self.get_recent_roi(real)
        dayno,slotno = self.get_slotno_parts()
        correct_dayno = self.reader.get_day_fname(dayno)
        current_rbr = self.remaining_budget_rate(correct_dayno)  # day_no is the correct one
        C = self.state_preserver.get_day_constraint(correct_dayno)
        rev_t = (self.state_preserver.rev_t if not real else self.state_preserver.rev_t_rw)

        # self.state_preserver.penalty_meter.add(gap)
        if slotno+1==self.num_slot_in_day:
            L_thresh = C
            b_thresh = 0
        else: # L_margin: the largest gap towards the target L
            dist_to_C = (self.L_margin if limit_param is None else limit_param) * (1 - (slotno + 1) / self.num_slot_in_day) ** 3 # decrease as time
            L_thresh = (1-dist_to_C)*C
            b_thresh = max((1-self.b_margin) * (1 - (slotno) / self.num_slot_in_day) ** 3 - 1. / self.num_slot_in_day, 0)  # need to have at least such many budget remaining

        # L_feasible = current_roi>=L_thresh
        # b_feasible = current_rbr>=b_thresh

        gap = min(0, current_roi-L_thresh) + min(0, current_rbr-b_thresh)
        rev_scale = self.get_rev_scaler(dayno) * 0.5  # normalize to [0,0.5]
        reward = rev_scale * rev_t
        rnet_r = reward if gap==0 else gap

        return rnet_r

    def indicator_approx(self, x):
        return torch.sigmoid(self.indicator_approx_v*x + self.sqrt_v)

    def compute_reward_powerlaw_differentiable(self, day_no, slot_no, limit_param=None, real=False):  # action: 0不变，1下调，2上调
        # don't use penalty scaler
        current_roi, last_roi = self.get_recent_roi(real)

        dayno, slotno = self.get_slotno_parts()

        correct_dayno = self.reader.get_day_fname(dayno)
        current_rbr = self.remaining_budget_rate(correct_dayno)  # day_no is the correct one
        C = self.state_preserver.get_day_constraint(correct_dayno)
        rev_t = (self.state_preserver.rev_t if not real else self.state_preserver.rev_t_rw)

        # self.state_preserver.penalty_meter.add(gap)
        if slotno + 1 == self.num_slot_in_day:
            L_thresh = C
            b_thresh = 0
            L_indicator_approx = 1. if current_roi>=L_thresh else 0.
        else:  # L_margin: the largest gap towards the target L
            dist_to_C = (self.L_margin if limit_param is None else limit_param) * (1 - (slotno + 1) / self.num_slot_in_day) ** 3  # decrease as time
            L_thresh = (1 - dist_to_C) * C
            b_thresh = max((1 - self.b_margin) * (1 - (slotno) / self.num_slot_in_day) ** 3 - 1. / self.num_slot_in_day,
                           0)  # need to have at least such many budget remaining
            L_indicator_approx = self.indicator_approx(current_roi - L_thresh) # maybe not approx well

        # L_feasible = current_roi>=L_thresh
        # b_feasible = current_rbr>=b_thresh
        # b_indicator_approx = self.indicator_approx(current_rbr-b_thresh)
        b_indicator = 1. if current_rbr>=b_thresh else 0.
        rev_scale = self.get_rev_scaler(dayno)
        reward = rev_scale * rev_t
        rnet_r = L_indicator_approx*b_indicator*reward + min(0, current_roi-L_thresh)*(1-L_indicator_approx) + + min(0, current_roi-b_thresh)*(1-b_indicator)
        return rnet_r

    def compute_reward_sparse(self, day_no, slot_no, alpha=None, real=False, soft=False): # action: 0不变，1下调，2上调
        # don't use penalty scaler
        # budget = self.state_preserver.get_day_budget(correct_dayno)
        # rev_t = (self.state_preserver.rev_t if not real else self.state_preserver.rev_t_rw)
        dayno, slotno = self.get_slotno_parts()
        if slotno+1==self.num_slot_in_day:
            current_roi, last_roi = self.get_recent_roi(real)
              # day_no is the correct one
            correct_dayno = self.reader.get_day_fname(dayno)
            C = self.state_preserver.get_day_constraint(correct_dayno)
            current_rbr = self.remaining_budget_rate(correct_dayno)
            # budget = self.get_day_budget(correct_dayno)
            # L_feasible = current_roi>=C
            # b_feasible = current_rbr>=0
            L_gap = current_roi - C # negative if infeasible
            b_gap = current_rbr - 0
            if soft:
                gap = (0 if alpha[0] is None else min(0,L_gap)*alpha[0]) + (0 if alpha[1] is None else min(b_gap,0)*alpha[1])
            else: gap = min(0, L_gap) + min(0, b_gap)

            rev_scale = self.get_rev_scaler(dayno) / self.num_slot_in_day # scale to [0,1]
            reward = rev_scale*self.state_preserver.rev_e
            if soft:
                rnet_r = reward + gap
            else: rnet_r = reward if gap==0 else gap
        else: rnet_r = 0
        return rnet_r

    def compute_reward_curiosity(self, day_no, slot_no, real=False):
        return self.compute_reward_sparse(day_no, slot_no, real=real, soft=False)*self.num_slot_in_day

    def compute_reward_shaped2(self, day_no, slot_no, real=False): # not for the budget setting
        sparse_reward = self.compute_reward_sparse(day_no, slot_no, real=real, soft=False)*self.num_slot_in_day
        dayno, slotno = self.get_slotno_parts()
        # current_roi, last_roi = self.get_recent_roi(real)
        # rev = (self.state_preserver.rev_e if not real else self.state_preserver.rev_e_rw)
        # gap, rev_e = self.core_reward_compute(current_roi, rev, C)
        # reward_scale = self.get_rev_scaler(dayno) # * self.num_slot_in_day
        if slotno!=self.num_slot_in_day-1:
            correct_dayno = self.reader.get_day_fname(dayno)
            C = self.state_preserver.get_day_constraint(correct_dayno)
            rev = self.state_preserver.rev_t if real else self.state_preserver.rev_t_rw
            rew = self.get_rev_scaler(dayno)*rev/2 # 48* slot_rev / oracle
            last_roi = self.state_preserver.last_step_ROI_e
            action = self.last_action_type

            if last_roi < C and action == 2: rew += -1
            if last_roi > C and action == 1: rew += -1
            return rew
        else: return sparse_reward

    def compute_reward_dense(self, real=False, combine_type=0, gap_type=0, accum=0, aux=0):
        """
        dense: reward for each slot
        piecewise:
        """
        current_roi, last_roi = self.get_recent_roi(real)
        dayno, slotno_in_day = self.get_slotno_parts()
        dayno, slotno = self.get_slotno_parts()
        correct_dayno = self.reader.get_day_fname(dayno)
        C = self.state_preserver.get_day_constraint(correct_dayno)
        # current_roi = rev / (cost + 1e-4)

        rev = (self.state_preserver.rev_t if not real else self.state_preserver.rev_t_rw) if accum==0 else (self.state_preserver.rev_e if not real else self.state_preserver.rev_e_rw)
        current_bcr = self.remaining_budget_rate(correct_dayno)
        if combine_type==0: # piecewise
            gap, rev = self.core_reward_compute(current_roi, rev, C, current_bcr)
        else: # soft combine
            gap = min(0, current_roi - C)

        self.state_preserver.penalty_meter.add(gap)

        if gap_type == 0: gap *= 1. # no scale (greedy)
        elif gap_type == 1: gap *= (slotno_in_day+1) * self.state_preserver.slot_scaler # linear decay
        else:
            factor = (self.gamma)**(self.num_slot_in_day-1 - slotno_in_day)
            gap *= factor # exp decay
        rev_scale = self.get_rev_scaler(dayno)
        rnet_r = rev_scale * rev + self.penalty_scale * gap
        # if slotno_in_day>43:
        #
        #     print(slotno_in_day, gap, factor, rnet_r)
        if aux>0:
            if slotno_in_day == self.num_slot_in_day - 1:
                if current_roi >= C: rnet_r = rev_scale*self.state_preserver.rev_e if not real else self.state_preserver.rev_e_rw
                else: rnet_r = self.penalty_scale*gap*self.num_slot_in_day
        return rnet_r


    def set_action_space(self, action_space):
        self.action_space = action_space

    def log_ratio_comparison(self, ratio_list, writer):
        dayno, slotno = (self.slot_no - 2) // self.num_slot_in_day, (self.slot_no - 2) % self.num_slot_in_day  #
        stepno = dayno*self.num_slot_in_day+slotno+1
        slot_oracle = self.get_slot_oracle_ratio(dayno, slotno)
        exact_ratio = ratio_list[0]
        writer.add_scalar('train_stats/exe_ratio_diff', exact_ratio-slot_oracle, global_step=stepno)

    def slot_step_cem(self, ratio_list, nothing, record_agent_ratio=True, do_train=False):
        # 1. take the corresponding day
        # 2. compute the results of applying the ratio to the current slot
        # 3. inc the slot, and prepare for the observation for the agent
        # 4. update some info
        # todo: cem for budget scenario
        state_preserver = self.state_preserver
        dayno, slotno = self.get_slotno_parts(past=False)
        has_rq, req = self.reader.get_day_slot_data(dayno, slotno)
        ret = [0] * 9
        bid_hr, bid_wkday, bid_dayno = None, None, None
        if has_rq: num_rq = len(req)
        else: num_rq = 0
        avg_value = 0.

        original_rewards, normalized_rewards = [], []
        revsum_act, costsum_act = 0, 0
        if has_rq:
            # req = curday_df[selector]
            num_sample_exp_upper = round(self.cem_exp_percent*num_rq)
            num_sample_exp = int(num_sample_exp_upper//self.cem_nsample * self.cem_nsample)
            num_sample_act = num_rq - num_sample_exp
            if num_sample_act==0 or num_sample_exp==0:
                has_rq = False

        if has_rq:
            # curday_df.loc[selector,'agent'] = np.empty(num_rq)
            ratios = np.empty(num_rq)
            indexes = np.random.permutation(np.arange(num_rq))
            sample_act_idxes = indexes[:num_sample_act]
            sample_exp_idxes = indexes[num_sample_act:]
            ratios[sample_act_idxes] = ratio_list[0]
            num_sample_in_each_batch = num_sample_exp // self.cem_nsample
            exp_ratios = [np.ones(num_sample_in_each_batch)*ratio_list[idx+1] for idx in range(self.cem_nsample)]
            ratios[sample_exp_idxes] = np.concatenate(exp_ratios)

            values = req['pctr'] * req['pppc']
            costs = req['costprice'] * 1e-3
            wins = ratios * values > costs
            # req.loc[wins, 'loose'] = ~wins
            fo_revsum = req.rev[wins].sum()
            fo_costsum = costs[wins].sum()
            fo_nwins = wins.sum()
            fo_nbids = len(req)

            ret = avg_value, fo_revsum, fo_costsum, fo_nwins, fo_nbids, fo_revsum, fo_costsum, fo_nwins, fo_nbids

            wins_act = wins.iloc[sample_act_idxes]
            revs_act = req.rev.iloc[sample_act_idxes]
            costs_act = costs.iloc[sample_act_idxes]
            revsum_act = revs_act[wins_act].sum()
            costsum_act = costs_act[wins_act].sum()
            # nwins_act = wins_act.sum()
            o_r, n_r = self.compute_reward_cem(revsum_act, costsum_act, num_sample_act)
            original_rewards.append(o_r)
            normalized_rewards.append(n_r)

            wins_exp = wins.values[indexes][num_sample_act:].reshape((-1,num_sample_in_each_batch))
            revs_exp = req.rev.values[indexes][num_sample_act:].reshape((-1,num_sample_in_each_batch))
            costs_exp = costs.values[indexes][num_sample_act:].reshape((-1,num_sample_in_each_batch))
            rsum_exp = (revs_exp*wins_exp).sum(-1)
            csum_exp = (costs_exp*wins_exp).sum(-1)
            tmp = [self.compute_reward_cem(rsum,csum,num_sample_in_each_batch) for rsum,csum in zip(rsum_exp, csum_exp)]
            original_rewards, normalized_rewards = list(zip(*tmp))
            original_rewards = [o_r]+list(original_rewards)
            normalized_rewards = [n_r]+list(normalized_rewards)

            # for idx in range(self.cem_nsample):
            #     sel = sample_exp_idxes[idx*num_sample_in_each_batch:(idx+1)*num_sample_in_each_batch]
            #     wins_exp = wins[sel]
            #     revs_exp = req.rev[sel]
            #     costs_exp = costs[sel]
            #     revsum_exp = revs_exp[wins_exp].sum()
            #     costsum_exp = costs_exp[wins_exp].sum()
            #     o_r, n_r = self.compute_reward_cem(revsum_exp, costsum_exp, num_sample_exp//self.cem_nsample)
            #     original_rewards.append(o_r)
            #     normalized_rewards.append(n_r)
        else:
            self.logger.info('something is wrong with this day')


        self.slot_no += 1
        state_preserver._update_rev_cost_slot(self.slot_no % self.num_slot_in_day, *ret)
        # state_preserver._update_action_slot(action, ratio_list[0])
        state_preserver._update_exe_revcost_slot(self.slot_no % self.num_slot_in_day, revsum_act, costsum_act)
        self.update_timestamp(bid_hr, bid_wkday, bid_dayno)


        if self.new_day_start: self.new_day_start = False
        if self.new_hour_start: self.new_hour_start = False

        # po_reward, po_reward_normalized, fo_reward, fo_reward_normalized = [None] * 4
        # if has_rq:
        #     fo_reward = self.compute_reward(None, None, real=True)  # reward of last slot
        #     fo_reward_normalized = state_preserver.normalize_reward(fo_reward,
        #                                                             real=True) if self.norm_reward else fo_reward
        #     if self.not_full_data():
        #         po_reward = self.compute_reward(None, None, real=False)  # reward of last slot
        #         po_reward_normalized = state_preserver.normalize_reward(po_reward,
        #                                                                 real=False) if self.norm_reward else po_reward
        #     else:
        #         po_reward, po_reward_normalized = fo_reward, fo_reward_normalized
        #
        #     state_preserver._update_reward_slot(po_reward, fo_reward)  # 必然落后，因为reward需要slot.cost才能计算
        return original_rewards, normalized_rewards

    def slot_step(self, action, ratio, execute_agent, multiplier=1., limit_parameter=None, record_agent_ratio=True):
        # 1. take the corresponding day
        # 2. compute the results of applying the ratio to the current slot
        # 3. inc the slot, and prepare for the observation for the agent
        # 4. update some info
        do_plot = self.do_plot
        state_preserver = self.state_preserver
        last_ratio = state_preserver.last_ratio
        if ratio-last_ratio<0: self.last_action_type = 1
        elif ratio-last_ratio>0: self.last_action_type = 2
        else: self.last_action_type = 0
        dayno, slotno = self.get_slotno_parts(past=False)
        has_rq, req = self.reader.get_day_slot_data(dayno, slotno)
        if record_agent_ratio: # should always update ratio, for yewu data some might not have every slot
            self.ratios.append(ratio)


        # if action < 3:
        #     self.last_action_type = 1
        # elif action > 3:
        #     self.last_action_type = 2
        # else:
        #     self.last_action_type = 0

        ret = [0]*8
        bid_hr, bid_wkday, bid_dayno = None, None, None
        step_reward = None
        avg_value = 0.

        if has_rq: # po is a subset of fo
            # req = curday_df.loc[selector]
            if False: # used to draw plots

                utility = req['pctr']
                delivery = req['rev']
                mprice = req['costprice']

                dfplot = pd.DataFrame(dict(mprice=mprice.values,
                                           delivery=delivery.values,
                                           utility=utility.values))
                valid_m = dfplot.mprice>0
                valid_d = dfplot.delivery>0
                valid_u = dfplot.utility>0
                # dfplot.columns = ['mprice','delivery','utility']
                # relative to python
                folder = '../dynamics_plots/{}_{}'.format(self.data_ver, 'test' if self.istest else 'train')
                os.makedirs(folder, exist_ok=True)

                ##################
                # d_m_joint = sns.jointplot(data=dfplot[['mprice','delivery']],
                #                           x='mprice', y='delivery', kind='kde',
                #                           joint_kws=dict(shade=True, thresh=0.01, cmap='Blues', zorder=0,
                #                           levels=[0.01, 0.25, 0.5, 0.75, 0.99]))
                toplot = dfplot[valid_m&valid_d][['mprice','delivery']]
                d_m_joint = sns.jointplot(data=toplot,
                                          x='mprice', y='delivery', kind='kde', xlim=(-5000,30000), ylim=(0,1000),
                                          joint_kws=dict(shade=True, cmap='Blues', zorder=0,
                                                         levels=[0.01, 0.25, 0.5, 0.75, 0.99]))
                d_m_joint.plot_marginals(sns.rugplot, color="r", height=-.1, clip_on=False)
                d_m_joint.plot_marginals(sns.histplot, bins=100)
                d_m_joint.savefig(os.path.join(folder, 'day{}_s{}_md.png'.format(dayno, slotno)))
                # import ipdb; ipdb.set_trace()
                #######################
                # toplot = dfplot[valid_m & valid_u][['mprice', 'utility']]
                # d_m_joint = sns.jointplot(data=toplot,
                #                           x='mprice', y='utility', kind='kde',
                #                           joint_kws=dict(shade=True, thresh=0.01, cmap='Blues', zorder=0,
                #                                          levels=[0.01, 0.25, 0.5, 0.75, 0.99]))
                # d_m_joint.plot_marginals(sns.rugplot, color="r", height=-.1, clip_on=False)
                # d_m_joint.plot_marginals(sns.histplot, bins=100)
                #
                # d_m_joint.savefig(os.path.join(folder, 'day{}_s{}_mu.png'.format(dayno, slotno)))


                # cost_scaler = self.distrs['cost_dynamics'][day][slotid] / costs.mean()
                # rev_scaler = self.distrs['rev_dynamics'][day][slotid] / ppcs.mean()
                # if slotid % 4 == 0 and do_plot:
                #     g = sns.jointplot(data=pd.DataFrame(dict(costprice=costs, rev=ppcs)),
                #                       x='costprice', y='rev', kind='kde',
                #                       joint_kws=dict(shade=True, thresh=0.01, cmap='Blues', zorder=0,
                #                                      levels=[0.01, 0.25, 0.5, 0.75, 0.99]))
                #     # g.plot_joint(sns.kdeplot, shade=True, thresh=0.01, cmap='Blues', zorder=0, levels=[0.01, 0.25, 0.5, 0.75, 0.99])
                #     g.plot_marginals(sns.rugplot, color="r", height=-.1, clip_on=False)
                #     g.plot_marginals(sns.histplot, bins=100)
                #     rounded_costscaler = round(cost_scaler, 2)
                #     rounded_revscaler = round(rev_scaler, 2)
                #     xlabel = g.ax_joint.get_xlabel()
                #     g.ax_joint.set_xlabel(xlabel + ' {}x'.format(rounded_costscaler))
                #     ylabel = g.ax_joint.get_ylabel()
                #     g.ax_joint.set_ylabel(ylabel + ' {}x'.format(rounded_revscaler))
                #     g.savefig('g{}.png'.format(slotid // 4))

            values = req['pctr'] * req['pppc']
            avg_value = values.mean()
            costs = req['costprice'] * 1e-3

            wins = ratio * values > costs
            # req.loc[wins, 'loose'] = ~wins
            fo_revsum = req.rev[wins].sum()
            fo_costsum = costs[wins].sum()
            fo_nwins = wins.sum()
            fo_nbids = len(req)
            if self.not_full_data():
                if self.loading=='slot': # compute loose arr
                    k2_ratio = self.reader.get_slot_oracle_ratio(None, slotno) # slot generation does not need dayno
                    partially_obs_selector = k2_ratio * values > costs
                else:
                    partially_obs_selector = req.loose==False

                po_wins = wins & partially_obs_selector
                po_revsum = req.rev[po_wins].sum()
                po_costsum = costs[po_wins].sum()
                po_nwins = po_wins.sum()
                po_nbids = len(req[partially_obs_selector].index)
                if self.use_syn():
                    syn_selector = partially_obs_selector
                    syn_values = values[syn_selector]
                    syn_costs = self.mprice_model.do_predict(req[req.loose]) * 1e-3
                    syn_wins = ratio * syn_values > syn_costs
                    po_revsum += syn_values[syn_wins].sum() # or pppc?
                    po_costsum += syn_costs[syn_wins].sum()
                    po_nwins += syn_wins.sum()
                    po_nbids = fo_nbids
                else: # directly copy what is seen

                    original_nwins = partially_obs_selector.sum()
                    wr = original_nwins / fo_nbids
                    po_wr = po_nwins / po_nbids if po_nbids>0 else 0.
                    po_nwins = round(po_wr*fo_nbids)
                    po_revsum /= (wr if wr>0 else 1.)
                    po_costsum /= (wr if wr>0 else 1.)
                    po_nbids = fo_nbids
            else:
                po_revsum, po_costsum, po_nwins, po_nbids = fo_revsum, fo_costsum, fo_nwins, fo_nbids

            bid_hr = req['hr'].iloc[0]

            ########### post process for budget scenario
            if self.budget>0 and (execute_agent or self.budget_braking):
                remaining_budget = state_preserver.remaining_budget
                if fo_costsum>remaining_budget:
                    po_revsum, po_costsum, po_nwins, fo_revsum, fo_costsum, fo_nwins = [0]*6
                    self.ratios[-1] = 0.
                    ratio = 0
            ret = po_revsum, po_costsum, po_nwins, po_nbids, fo_revsum, fo_costsum, fo_nwins, fo_nbids

        self.slot_no += 1
        state_preserver._update_rev_cost_slot(self.slot_no % self.num_slot_in_day, avg_value, *ret)
        state_preserver._update_action_slot(action, ratio)
        self.update_timestamp(bid_hr, bid_wkday, bid_dayno)

        if self.new_day_start: self.new_day_start = False
        if self.new_hour_start: self.new_hour_start = False

        po_reward, po_reward_normalized, fo_reward, fo_reward_normalized = [None]*4
        if has_rq:
            fo_reward = self.compute_reward(self.reader.get_day_fname(dayno), None, multiplier, limit_parameter, real=True)  # reward of last slot
            fo_reward_normalized = state_preserver.normalize_reward(fo_reward, real=True) if self.norm_reward else fo_reward
            if self.not_full_data():
                po_reward = self.compute_reward(self.reader.get_day_fname(dayno), None, multiplier, limit_parameter, real=False)  # reward of last slot
                po_reward_normalized = state_preserver.normalize_reward(po_reward, real=False) if self.norm_reward else po_reward
            else: po_reward, po_reward_normalized = fo_reward, fo_reward_normalized

            state_preserver._update_reward_slot(po_reward, fo_reward) # 必然落后，因为reward需要slot.cost才能计算
        return po_reward, po_reward_normalized

    def slot_step_uscb(self, action, ratio, execute_agent, record_agent_ratio=True, do_train=False):

        state_preserver = self.state_preserver
        last_ratio = state_preserver.last_ratio
        if ratio-last_ratio<0: self.last_action_type = 1
        elif ratio-last_ratio>0: self.last_action_type = 2
        else: self.last_action_type = 0
        dayno, slotno = self.get_slotno_parts(past=False)
        has_rq, req = self.reader.get_day_slot_data(dayno, slotno)
        if record_agent_ratio: # should always update ratio, for yewu data some might not have every slot
            self.ratios.append(ratio)

        ret = [0]*8
        bid_hr, bid_wkday, bid_dayno = None, None, None
        step_reward = None
        avg_value = 0.
        fo_revsum = 0
        won_rsum, won_csum = None, None
        if has_rq: # po is a subset of fo
            ratios = np.asarray(self.state_preserver.ratio_list+[ratio])

            values = req['pctr'] * req['pppc']
            bids = np.outer(ratios, values.values) # different ratios, nreq
            avg_value = values.mean()
            costs = req['costprice'].values * 1e-3
            # print(len(req))
            if np.isnan(bids).any() or np.isnan(costs).any():
                import ipdb; ipdb.set_trace()
            wins = (bids > costs[None, :])
            winsum = wins.sum(1)

            won_rsum = np.einsum('mn,n->m', wins, req.rev.values)
            won_csum = np.einsum('mn,n->m', wins, costs)
            # req.loc[wins, 'loose'] = ~wins
            fo_revsum = won_rsum[-1]
            fo_costsum = won_csum[-1]
            fo_nwins = winsum[-1]
            fo_nbids = len(req)
            if self.not_full_data():
                if self.loading=='slot': # compute loose arr
                    k2_ratio = self.reader.get_slot_oracle_ratio(None, slotno) # slot generation does not need dayno
                    partially_obs_selector = k2_ratio * values > costs
                else:
                    partially_obs_selector = req.loose==False
                # wins in won datasets

                po_wins = wins & partially_obs_selector
                po_revsum = req.rev[po_wins].sum()
                po_costsum = costs[po_wins].sum()
                po_nwins = po_wins.sum()
                po_nbids = len(req[partially_obs_selector].index)
                if self.use_syn():
                    syn_selector = partially_obs_selector
                    syn_values = values[syn_selector]
                    syn_costs = self.mprice_model.do_predict(req[req.loose]) * 1e-3
                    syn_wins = ratio * syn_values > syn_costs
                    po_revsum += syn_values[syn_wins].sum() # or pppc?
                    po_costsum += syn_costs[syn_wins].sum()
                    po_nwins += syn_wins.sum()
                    po_nbids = fo_nbids
                else: # directly copy what is seen

                    original_nwins = partially_obs_selector.sum()
                    wr = original_nwins / fo_nbids
                    po_wr = po_nwins / po_nbids if po_nbids>0 else 0.
                    po_nwins = round(po_wr*fo_nbids)
                    po_revsum /= (wr if wr>0 else 1.)
                    po_costsum /= (wr if wr>0 else 1.)
                    po_nbids = fo_nbids
            else:
                po_revsum, po_costsum, po_nwins, po_nbids = fo_revsum, fo_costsum, fo_nwins, fo_nbids

            # req_wloose = self.bid_requests[selector]
            bid_hr = req['hr'].iloc[0]
            ########### post process for budget scenario
            if self.budget > 0 and execute_agent:
                remaining_budget = state_preserver.remaining_budget
                if fo_costsum > remaining_budget and len(self.ratios)>0:
                    po_revsum, po_costsum, po_nwins, fo_revsum, fo_costsum, fo_nwins = [0] * 6
                    self.ratios[-1] = 0.
                    ratio = 0.

            ret = po_revsum, po_costsum, po_nwins, po_nbids, fo_revsum, fo_costsum, fo_nwins, fo_nbids

        self.slot_no += 1
        state_preserver._update_rev_cost_slot(self.slot_no % self.num_slot_in_day, avg_value, *ret)
        state_preserver._update_action_slot(action, ratio)
        self.update_timestamp(bid_hr, bid_wkday, bid_dayno)

        if self.new_day_start: self.new_day_start = False
        if self.new_hour_start: self.new_hour_start = False

        po_reward, po_reward_normalized, fo_reward, fo_reward_normalized = [None]*4
        if has_rq:
            fo_reward = fo_revsum # reward of last slot
            fo_reward_normalized = state_preserver.normalize_reward(fo_reward, real=True) if self.norm_reward else fo_reward
            if self.not_full_data():
                po_reward = self.compute_reward(self.reader.get_day_fname(dayno), None, real=False)  # reward of last slot
                po_reward_normalized = state_preserver.normalize_reward(po_reward, real=False) if self.norm_reward else po_reward
            else: po_reward, po_reward_normalized = fo_reward, fo_reward_normalized

            state_preserver._update_reward_slot(po_reward, fo_reward) # 必然落后，因为reward需要slot.cost才能计算
        return won_rsum, won_csum

    def log_episode_perf(self, perf, k2_info, output_dir): # todo use realworld

        perf = perf + k2_info
        pd.DataFrame([perf]).to_csv(os.path.join(output_dir, 'results.csv'), header=False, index=False, mode='a')

    def get_day_fname(self):
        dayno = (self.slot_no-1) // self.num_slot_in_day
        return self.reader.get_day_fname(dayno)

    def step_logging(self, during_exp, state_preserver, original_r, normalized_r, ratio, output_dir, model_name):
        """
        log to actions.csv
        """
        # model_name: get_model_name
        slot_no = (self.slot_no - 1) % self.num_slot_in_day
        dayno = (self.slot_no - 1) // self.num_slot_in_day  # past day
        dayno_fname = self.get_day_fname()
        # df = self.reader.get_day_data(dayno) #self.bid_requests_list[dayno]#self.oracle_ratios[dayno]
        ref_ratio = self.get_slot_oracle_ratio(dayno, slot_no)

        # log the results
        roi = state_preserver.get_overall_roi()
        basics = state_preserver.get_basics()
        step_rev = basics[0]
        slot_no = self.get_current_slot()
        dat = [self.spec.id, 'exp' if during_exp else 'act',
               slot_no,
               dayno_fname,
               ratio,
               roi,
               original_r,
               normalized_r,
               ref_ratio,
               state_preserver.get_step_wr(),
               ] + basics + [original_r, roi, min(roi-1,0)*self.penalty_scale, step_rev, step_rev*self.reward_scale, min(roi-1,0)*0.95**(95-slot_no)]

        pd.DataFrame([dat]).to_csv(os.path.join(output_dir, '{}_actions.csv'.format(model_name)),
                                   header=False,
                                   index=False, mode='a')

    def dayover_logging(self, is_eval, state_preserver, output_dir):
        dayno = (self.slot_no - 1) // self.num_slot_in_day
        self.log_episode_perf(state_preserver.get_perf(dayno, 'test' if self.istest else ('eval' if is_eval else 'train')),
                          self.get_k2_info(), output_dir)
        state_preserver.return_meter.add(state_preserver.rev_e)
        avg_rev_scale = state_preserver.return_meter.value()[0]
        avg_pen_scale = state_preserver.penalty_meter.value()[0]
        self.logger.info('the following is day {} with budget {} L {}: average reward scaler, {}, average penalty, {} '.format(
            self.get_day_fname(),
            self.get_curday_budget(), #self.budgets[self.reader.get_day_fname(dayno)] if self.budgets is not None else None,
            self.get_curday_constraint(),
            self.state_preserver.reward_meter_rw.get_std(),
            avg_pen_scale))

        if len(state_preserver.avg_scale_records) > 0 and len(state_preserver.avg_scale_records)%3==0:
            factor = avg_rev_scale / np.mean(state_preserver.avg_scale_records[-6:])
            is_soft_reward = self.reward_type>=9 and self.get_reward_sub_types()[0]==1
            if not is_soft_reward: factor = 1.
            self.penalty_scale = self.penalty_scale * factor
            self.logger.info('penalty factor {}, current penalty scale {}. (soft-reward:{})'.format(factor, self.penalty_scale, is_soft_reward))
        state_preserver.avg_scale_records.append(avg_rev_scale)
        self.day_valuesum.append(state_preserver.avg_value_sum)

    def weekover_logging(self, during_exp, state_preserver, ):
        if not during_exp:
            avg_rew_scale = state_preserver.return_meter.value()[0]
            avg_pen_scale = state_preserver.penalty_meter.value()[0]
            self.logger.info('week over: average reward scale, {}, average penalty, {} '.format(avg_rew_scale,
                                                                                               avg_pen_scale))

            state_preserver.return_meter.reset()
            state_preserver.penalty_meter.reset()


    def get_observation_cem(self, extra=None):
        """
        return: list of bins
        """
        # state_preserver = self.state_preserver
        dayno, slotno = self.get_slotno_parts(past=False)
        if not self.is_syn: # for real data, need to check if exist current slot
            curday_df = self.reader.get_day_data(dayno) #self.bid_requests_list[self.cur_day]
            tslot = self.slot_no % self.num_slot_in_day
            selector = curday_df.time_slot == tslot
            has_rq = selector.sum()
            if has_rq:
                obs = [tslot] * (1+self.cem_nsample) # act, exp each
            else:
                obs = None
        else:
            obs = [slotno] * (1 + self.cem_nsample)

        return obs

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass


