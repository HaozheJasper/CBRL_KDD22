import pandas as pd
import os
import sys
import torch
import configparser
import numpy as np
import random
import time
import logging
import gym
import pickle as pkl
from collections import deque
sys.path.append(os.getcwd()+'/src')
# import auction_simulator
from auction_simulator import pre_run_slot_generator
import itertools

class interfaces():
    def all_over(self, *args):
        pass

    def set_day(self, day_cnt):
        self.cur_day = day_cnt
        # if not self.is_train():
        #     print('day {} over, next day is eval iter'.format(day_cnt-1))
        # else: print('day {} over, next day is train iter'.format(day_cnt-1))

    def set_execution(self, mode=True):
        self.execution = mode

    def during_exp(self):
        # import ipdb; ipdb.set_trace()
        return not self.execution

    def postprocess_reward(self, obs, ratio, nobs, rew):
        return rew

    def get_multiplier(self):
        return 1.

    def get_limit_parameter(self):
        return None

    def get_bid_ratio(self, last_ratio, action):
        action = action.item()
        if self.ratio_update_type == 0: # independent update
            return action
        else: return action + last_ratio

data_seeds = list(range(80))
# deprecated
class Checkpointer():
    def __init__(self, loss_length=1000, acc_length=100, min_length_ratio=0.3, is_ep=False, old_yewu=False):
        # loss status, acc status
        self.is_ep = is_ep
        self.loss_length = loss_length
        self.loss_q = deque(maxlen=loss_length)
        self.acc_q = deque(maxlen=acc_length)
        self.min_loss_len = round(min_length_ratio*loss_length)
        self.min_acc_len = 15 # round(min_length_ratio*acc_length)
        self.lower_quantile = 0.1
        self.upper_quantile = 0.9
        # self.history_lower_quants = []
        # self.history_upper_quants = []
        self.history_median = []
        self.loss_iters = 0

        self.acc_quantile = 0.95
        self.history_mean_acc = []
        self.acc_iters = 0
        self.acc_update_freq = 10
        self.max_perf = -np.inf
        self.old_yewu = old_yewu

    def add_loss_item(self, item):
        self.loss_q.append(item)
        self.loss_iters += 1
        if self.loss_iters%self.loss_length==0:
            self.history_median.append(np.median(self.loss_q))
            # self.history_upper_quants.append(np.quantile(self.loss_q, self.upper_quantile))
            # self.history_lower_quants.append(np.quantile(self.loss_q, self.lower_quantile))

    def add_acc_item(self, item):
        self.acc_q.append(item)
        self.acc_iters += 1
        if self.acc_iters%self.acc_update_freq==0:
            self.history_mean_acc.append(np.mean(list(self.acc_q)[-self.min_acc_len:]))

    def check_length(self):
        msg = ''
        if len(self.loss_q)>self.min_loss_len and len(self.acc_q)>self.min_acc_len:
            return True,msg
        else:
            msg = 'lengths: loss {}, acc {}'.format(len(self.loss_q), len(self.acc_q))
            # print()
            return False,msg

    def check_length2(self):
        msg = ''
        if len(self.acc_q)>15:
            return True,msg
        else:
            msg = 'lengths: loss {}, acc {}'.format(len(self.loss_q), len(self.acc_q))
            # print()
            return False,msg

    def is_loss_stable(self):
        if len(self.loss_q)==0: return False,''
        if self.loss_q[-1]<10.: return True,''
        median = np.median(self.loss_q)
        if len(self.history_median)<=3:
            median_ok = True
            hi = None
            whisker = None
        else:
            hi = np.quantile(self.history_median, self.upper_quantile)
            whisker = hi - np.quantile(self.history_median, self.lower_quantile)
            median_ok = median<(hi+1.5*whisker)
        bar = np.quantile(self.loss_q, self.upper_quantile)
        msg = ''
        if bar<10.: last_ok = True
        else:
            last_ok = self.loss_q[-1]<bar

        if median_ok and last_ok:
            return True, msg
        else:
            msg = 'median {}, hi {}, whisker {}, last {}, bar {}'.format(median, hi, whisker, self.loss_q[-1], bar )
            return False, msg

    def is_loss_stable2(self):
        return True, ''

    def is_acc_ok(self):
        # 窗口内稳定的前提下
        confidence_on_len = len(self.acc_q)>=15
        if not confidence_on_len: return False,'acc list not reliable with length {}'.format(len(self.acc_q))
        acc_list = list(self.acc_q)
        msg = ''
        if len(self.history_mean_acc)>=2:
            bar = np.quantile(self.history_mean_acc, 0.8)

            running_mean = np.mean(acc_list[-self.min_acc_len:])
            is_promising = running_mean>bar
            if not is_promising:
                msg += 'running mean < bar: {},{}\n'.format(running_mean, bar)

        else: is_promising = True

        if is_promising:
            acc_list = list(self.acc_q)
            latest_mean = np.mean(acc_list[-8:])
            last_mean = np.mean(acc_list[-16:-8])
            if latest_mean > last_mean: return True,''
            else:
                msg += 'latest vs last: {} {}'.format(latest_mean, last_mean)

        return False, msg

    def is_acc_ok2(self):
        acc_list = list(self.acc_q)
        running_mean = np.mean(acc_list[-8:])
        bar = np.quantile(acc_list, 0.5)
        if running_mean>bar and acc_list[-1]>bar:
            return True,''
        else: return False,'running mean over 8: {}, bar {}, last {}'.format(running_mean, bar, acc_list[-1])

    def is_acc_ok3(self):
        acc_list = list(self.acc_q)
        running_mean = np.mean(acc_list[-30:])
        if running_mean>self.max_perf-0.1:
            if running_mean>self.max_perf: self.max_perf = running_mean
            return True, ''
        else: return False, 'running mean over 15: {}, max perf {}'.format(running_mean, self.max_perf)


    def check_save(self):
        if self.is_ep or self.old_yewu:
            lenok, msg1 = self.check_length2()
            lossok, msg2 = self.is_loss_stable2()
            accok, msg3 = self.is_acc_ok2() if self.is_ep else self.is_acc_ok3()
            return lenok and lossok and accok, '\n'.join([msg1, msg2, msg3])
        else:
            lenok,msg1 = self.check_length()
            lossok,msg2 = self.is_loss_stable()
            accok,msg3 = self.is_acc_ok()
            return lenok and lossok and accok, '\n'.join([msg1,msg2,msg3])

def get_logger(output_dir):
    logger = logging.getLogger('default')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.DEBUG)
    streamhandler.setFormatter(formatter)
    filehandler = logging.FileHandler(os.path.join(output_dir, 'log.txt'))
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    logger.addHandler(filehandler)
    return logger

def compute(name, rev, cost, win, numbid, reward, exe_rev=None, exe_cost=None, other_rev=dict(), constraints=(1.,), margin=0.01):
    C = constraints[0]
    tmp = dict(rev=rev, cost=cost, roi=rev/(1e-4+cost), exe_roi=exe_rev/(1e-4+exe_cost) if exe_rev is not None else 0., win=win, winrate=win/numbid, reward=reward)
    for k,v in other_rev.items():
        imp = round((rev / v - 1)*100,2)
        tmp['{}_rev_imp_percent'.format(k)] = imp if tmp['roi'] > C else np.nan
        tmp['{}_rev_imp_percent_soft'.format(k)] = imp if tmp['roi'] > C - margin else np.nan
        tmp['{}_rev_imp_percent_raw'.format(k)] = imp

    return name, tmp

def upper_bounded(ar, kr, mode, upbound=0.02):
    return min(ar / kr - 1., upbound) if mode in ['oracle','rev_oracle'] else ar / kr - 1.

def compute2(env,k, window_size, comp=dict(), other_rois=dict(), margin=None):

    rev_list, normalized_rev_list, cost_list, reward_list, roi_err_list, oracle_list, constraint_list \
        = env.reference_all_revs[k], \
          env.reference_all_revs[k + '_normalized'],\
          env.reference_all_costs[k],\
          env.reference_all_rewards[k],\
          env.reference_all_roi_err[k],\
          env.all_oracle, \
          env.all_constraint

    mcb_scores = env.montecarlo_soft_reward(np.asarray(rev_list),
                                            np.asarray(cost_list),
                                            np.asarray(oracle_list),
                                            C=np.asarray(constraint_list),
                                            gamma=3)
    if margin is None: margin = 0.01
    # roi_list = [r/(c+1e-4) for r,c in zip(rev_list,cost_list)]
    sat = [x>=0 for x in roi_err_list]
    sat_soft = [x>=-margin for x in roi_err_list]
    roi_gap_soft = [x if x<-margin else 0. for x in roi_err_list] # todo: should indeed mult by 100
    # window_roi_list = roi_list[-window_size:]
    num_sat_window = sum(sat[-window_size:])
    num_sat_soft_window = sum(sat_soft[-window_size:])
    num_sat = sum(sat)
    num_sat_soft = sum(sat_soft)
    # roi sat:
    window_sat_rate = num_sat_window / window_size
    window_sat_soft_rate = num_sat_soft_window / window_size
    sat_rate = num_sat / len(rev_list)

    sat_soft_rate = sum(sat_soft) / len(rev_list)
    # avg rev:
    rev_cond = [r for s,r in zip(sat, rev_list) if s]
    rev_cond_soft = [r for s,r in zip(sat_soft, rev_list) if s]
    rev_weight = [r*s for s,r in zip(sat, rev_list)]
    rev_weight_soft = [r*s for s,r in zip(sat_soft, rev_list)]
    accum_rev_cond = sum(rev_cond)
    accum_rev_cond_soft = sum(rev_cond_soft)
    avg_rev_cond = sum(rev_cond) / num_sat_window if num_sat_window>0 else np.nan # 只看满足
    avg_rev_cond_soft = sum(rev_cond_soft) / num_sat_soft_window if num_sat_soft_window else np.nan
    window_accum_rev_cond = sum([x for x in rev_weight[-window_size:] if x>0])
    window_accum_rev_cond_soft = sum([x for x in rev_weight_soft[-window_size:] if x>0])
    window_avg_rev_cond = window_accum_rev_cond/num_sat_window if num_sat_window>0 else np.nan
    window_avg_rev_cond_soft = window_accum_rev_cond_soft/num_sat_soft_window if num_sat_soft_window>0 else np.nan
    accum_rev = sum(rev_weight)
    accum_rev_soft = sum(rev_weight_soft)
    avg_rev = accum_rev / len(rev_weight)
    avg_rev_soft = accum_rev_soft / len(rev_weight_soft)
    window_accum_rev = sum(rev_weight[-window_size:])
    window_accum_rev_soft = sum(rev_weight_soft[-window_size:])
    window_avg_rev = window_accum_rev / window_size
    window_avg_rev_soft = window_accum_rev_soft / window_size
    window_avg_rew = sum(reward_list[-window_size:]) / float(window_size)
    accum_avg_rew = sum(reward_list) / float(len(reward_list))
    normalized_window_avg_rev_soft = sum([gap if gap<0 else r for gap,r in zip(roi_gap_soft[-window_size:], normalized_rev_list[-window_size:])])/window_size
    normalized_window_avg_rev = sum([r * s for s, r in zip(sat[-window_size:], normalized_rev_list[-window_size:])]) / window_size

    tmp = dict(normalized_window_avg_rev_soft=normalized_window_avg_rev_soft,
               normalized_avg_rev_soft=roi_gap_soft[-1] if roi_gap_soft[-1]<0 else normalized_rev_list[-1],
               normalized_window_avg_rev=normalized_window_avg_rev,
        window_avg_rev=window_avg_rev,
        window_soft_avg_rev=window_avg_rev_soft,
         window_cond_avg_rev=window_avg_rev_cond,
        window_soft_cond_avg_rev=window_avg_rev_cond_soft,
         window_roi_sat=window_sat_rate,
        window_roi_soft_sat=window_sat_soft_rate,
         window_avg_rew=window_avg_rew,
         accum_avg_rev=avg_rev,
        accum_avg_rev_soft=avg_rev_soft,
         accum_cond_avg_rev=avg_rev_cond,
        accum_soft_cond_avg_rev=avg_rev_cond_soft,
         accum_roi_sat=sat_rate,
        accum_soft_roi_sat=sat_soft_rate,
        accum_avg_rew=accum_avg_rew)

    for k,vlist in comp.items(): # vlist: revlist
        other_roi_err = other_rois.get(k, None)
        if other_roi_err is None:
            other_roi_err = [0]*len(vlist)
        window_cond_imp = [upper_bounded(ar, kr, mode=k) for ar,kr,r in zip(rev_weight[-window_size:],vlist[-window_size:],other_roi_err[-window_size:]) if ar>0 and r>=0] # ar>0, satisfied
        window_cond_avg_imp = np.mean(window_cond_imp) if len(window_cond_imp)>0 else np.nan
        # window_avg_imp = sum([(ar/kr - 1.)*ar for ar,kr in zip(rev_weight[-window_size:],vlist[-window_size:]) if ar>0])  / num_sat_window if num_sat_window>0 else np.nan
        window_soft_cond_imp = [upper_bounded(ar, kr, mode=k) for ar, kr, r in zip(rev_weight_soft[-window_size:], vlist[-window_size:], other_roi_err) if ar > 0 and r>=-margin]
        window_soft_cond_avg_imp = np.mean(window_soft_cond_imp) if len(window_soft_cond_imp)>0 else np.nan
        tmp['{}_window_cond_avg_imp'.format(k)] = round(window_cond_avg_imp*100,2)
        tmp['{}_window_soft_cond_avg_imp'.format(k)] = round(window_soft_cond_avg_imp*100, 2)
        accum_cond_imp = [upper_bounded(ar, kr, mode=k) for ar, kr, r in zip(rev_weight, vlist, other_roi_err) if ar > 0 and r>=0]
        accum_cond_avg_imp = np.mean(accum_cond_imp) if len(accum_cond_imp)>0 else np.nan
        tmp['{}_accum_cond_avg_imp'.format(k)] = round(accum_cond_avg_imp * 100, 2)
        accum_soft_cond_imp = [upper_bounded(ar, kr, mode=k) for ar, kr, r in zip(rev_weight_soft, vlist, other_roi_err) if ar > 0 and r>=-margin]
        accum_soft_cond_avg_imp = np.mean(accum_soft_cond_imp) if len(accum_soft_cond_imp)>0 else np.nan
        tmp['{}_accum_soft_cond_avg_imp'.format(k)] = round(accum_soft_cond_avg_imp * 100, 2)

    normalized_imps = []
    cond_imps = []
    seq = rev_weight_soft[-window_size:]
    seq_oracle = oracle_list[-window_size:]
    length = len(seq)
    for idx,weighted_rev in enumerate(seq):
        if weighted_rev>0:
            imp = min((weighted_rev/seq_oracle[idx])*100,104)
            imp = round(imp, 2)
            normalized_imps.append(imp)
            cond_imps.append((imp-100))
        else: normalized_imps.append(0) # 不满足是0，满足应该比0高

    tmp['accum_avg_norm_imp'] = np.mean(normalized_imps) if length>0 else np.nan
    tmp['accum_cond_imp'] = np.nanmean(cond_imps) if len(cond_imps)>0 else np.nan
    tmp['normalized_imp_soft'] = normalized_imps[-1] if len(normalized_imps)>0 else np.nan
    tmp['accum_gscore'] = np.mean(mcb_scores) if length>0 else np.nan
    return tmp

def get_tags(k):
    if 'accum' in k: return 'daily_accum'
    elif 'window' in k: return 'window'
    else: return 'daily_single'

def write_tb(agent_info, keys, writer, prefix, method_name, logtype, step, agent_metrics=dict()):
    """
        daily-single: agent_info
        daily-accum: agent_metrics, accum
        window: agent_metrics, window
    """
    stats = agent_info[1]
    agent_roi = stats['roi']
    stats.update(agent_metrics)
    # agent_info = stats
    for k in keys:
        if k not in stats: continue
        if stats[k] is None: continue # some does not have reward
        writer.add_scalar('{}_{}/{}/{}'.format(prefix, get_tags(k), method_name, k), stats[k], step)

    writer.add_scalar('{}_{}/{}/{}'.format(prefix,  'daily_single' , method_name,'roi_hurt'), (min(agent_roi,1.)-1)*100, step)

def report(title, env, agent, output_dir, window_size=None, exe_rev=None, exe_cost=None, writer=None, logger=None, base_day=0, logtypes_elim=None, log_as_final=False, max_perf=-np.inf, margin=0.02):
    """
    daily-single: when ndays==1
    daily-accum: when ndays>1
    window: when ndays>1

    """
    # margin = 0.02
    ndays = 1
    num_bid = sum(env.all_bids['oracle'][-ndays:])
    logtemplate = '{}: rev={rev}, cost={cost}, roi={roi}, win={win}, winrate={winrate}'
    refs = ['slot_oracle', 'oracle']
    if not env.is_syn:
        refs.append('k2')

    keys = ['rev','cost','roi','exe_roi', 'win','winrate','reward',
            'window_avg_rev', 'window_soft_avg_rev', 'accum_avg_rev', 'accum_avg_rev_soft','accum_avg_rew',
            'window_cond_avg_rev', 'window_soft_cond_avg_rev', 'accum_cond_avg_rev', 'accum_soft_cond_avg_rev',
            'window_roi_sat', 'window_roi_soft_sat', 'accum_roi_sat', 'accum_soft_roi_sat', 'window_avg_rew',
            'normalized_window_avg_rev_soft', 'normalized_window_avg_rev', 'accum_avg_norm_imp', 'accum_gscore', 'accum_cond_imp'
             ] + ['{}_rev_imp_percent'.format(k) for k in refs] \
           + ['{}_rev_imp_percent_soft'.format(k) for k in refs] \
           + ['{}_rev_imp_percent_raw'.format(k) for k in refs] \
           + ['{}_window_cond_avg_imp'.format(k) for k in refs] \
           + ['{}_window_soft_cond_avg_imp'.format(k) for k in refs] \
           + ['{}_accum_cond_avg_imp'.format(k) for k in refs] \
           + ['{}_accum_soft_cond_avg_imp'.format(k) for k in refs]
    log_keys = ['accum_avg_rev_soft', 'accum_avg_rew', 'accum_soft_cond_avg_rev', # rev or reward
                'accum_soft_roi_sat',
                '{}_accum_soft_cond_avg_imp'.format('oracle' if env.is_syn else 'slot_oracle'), # yewu data slot oracle is better
                '{}_accum_soft_cond_avg_imp'.format('slot_oracle' if env.is_syn else 'k2'),
                'accum_avg_norm_imp', 'accum_gscore', 'accum_cond_imp'
                ]
    to_compare = ['slot_oracle', 'oracle', 'k2']
    other_revs = {k:env.reference_all_revs[k] for k in to_compare if k in env.reference_keys}
    other_costs = {k:env.reference_all_costs[k] for k in to_compare if k in env.reference_keys}
    other_roi_errors = {k:np.asarray(other_revs[k])/(np.asarray(other_costs[k])+1e-4)-np.asarray(env.all_constraint) for k in other_revs}

    if logtypes_elim is not None: # daily or window
        if logtypes_elim=='daily': # don't include daily
            keys = [k for k in keys if 'window' in k]
        if logtypes_elim=='window':
            keys = [k for k in keys if 'window' not in k]
    prefix = env.mode
    step = base_day
    # print('stats logging step, {}'.format(step))
    # if logtypes_elim=='window':
    #     print(step, env.get_day(), base_day)
    logtype = 'daily' if ndays==1 else 'window'
    k = 'agent'
    # compute daily, stats + compare
    agent_info = compute('agent_real',
                         sum(env.reference_all_revs[k][-ndays:]),
                         sum(env.reference_all_costs[k][-ndays:]),
                         sum(env.reference_all_wins[k][-ndays:]),
                         num_bid,
                         sum(env.reference_all_rewards[k][-ndays:]),
                         other_rev={k:v[-1] for k,v in other_revs.items()},
                         constraints=env.all_constraint[-ndays:],
                         exe_rev=exe_rev,
                         exe_cost=exe_cost) # incorrect for test
    # compute sequence
    agent_metrics = compute2(env,k,
                             min(window_size, len(env.reference_all_revs[k])),
                             comp=other_revs,
                             other_rois=other_roi_errors,
                             margin=margin,

                         )
    # else: agent_metrics = []

    agent_rev, agent_roi = agent_info[1]['rev'], agent_info[1]['roi']
    logger.info('='*40 + title + '='*40)
    logger.info('num_bid={}'.format(num_bid))

    logger.info(logtemplate.format(agent_info[0], **agent_info[1]))
    method_name = agent_info[0]
    write_tb(agent_info, keys, writer, prefix, method_name, logtype, step, agent_metrics=agent_metrics)
    # log performance

    if log_as_final:
        tmp = output_dir.split(os.path.sep)
        family_folder = os.path.sep.join(tmp[:-1])
        subfolder = tmp[-1]
        d = dict(expname=subfolder, train_window_avg_rew=max_perf)
        d.update({k: agent_info[1][k] for k in log_keys})
        df = pd.DataFrame(d, index=[0])
        # ['expname', 'train_window_avg_rew', 'accum_avg_rev_soft',
        #  'accum_avg_rew', 'accum_soft_cond_avg_rev', 'accum_soft_roi_sat',
        #  'oracle_accum_soft_cond_avg_imp', 'slot_oracle_accum_soft_cond_avg_imp',
        #  'accum_avg_norm_imp']
        df.to_csv(os.path.join(family_folder, 'perf.csv'), index=False, header=False, mode='a')
        logger.info('perf.csv dumped')

    if env.not_full_data():
        k = 'agent_po'
        nbid2 = sum(env.all_bids['agent_po'][-ndays:])
        agent_info = compute('agent_po',
                             sum(env.reference_all_revs[k][-ndays:]),
                             sum(env.reference_all_costs[k][-ndays:]),
                             sum(env.reference_all_wins[k][-ndays:]),
                             nbid2,
                             sum(env.reference_all_rewards[k][-ndays:]),
                             other_rev={k:v[-1] for k,v in other_revs.items()},
                             constraints=env.all_constraint[-ndays:],
                             exe_rev=exe_rev,
                             exe_cost=exe_cost)  # incorrect for test
        # window/accum 一起算了
        agent_metrics = compute2(env,k,
                                 min(window_size, env.get_record_length()),
                                 comp=other_revs,
                                 other_rois=other_roi_errors,
                                 margin=margin,
                                 )
        # else: agent_metrics = []

        # agent_rev, agent_roi = agent_info[1]['rev'], agent_info[1]['roi']
        logger.info('=' * 40 + title + '=' * 40)
        logger.info('num_bid={}'.format(nbid2))

        logger.info(logtemplate.format(agent_info[0], **agent_info[1]))
        method_name = agent_info[0]
        write_tb(agent_info, keys, writer, prefix, method_name, logtype, step, agent_metrics=agent_metrics)


    logger.info('hurt roi constraint: {:.2f}%'.format(
        min(agent_roi-env.all_constraint[-1], 0) * 100
    ))
    logger.info('-' * 84)

    for k in env.reference_keys[:2]: # make sure top 2 are k2 and oracle
        env_info = compute(k,
                           sum(env.reference_all_revs[k][-ndays:]),
                           sum(env.reference_all_costs[k][-ndays:]),
                           sum(env.reference_all_wins[k][-ndays:]),
                           num_bid,
                           None,
                           other_rev=dict())
        k2_rev, k2_roi = env_info[1]['rev'], env_info[1]['roi']
        logger.info(logtemplate.format(env_info[0], **env_info[1]))
        # accum stats
        revs = env.reference_all_revs[k]
        numdays = len(env.reference_all_revs[k])
        roi_errors = [env.reference_all_revs[k][i]/(env.reference_all_costs[k][i]+1e-4)-env.all_constraint[i] for i in range(numdays)]
        roi_soft_sat = [x>=-margin for x in roi_errors ]
        cond_revs = [r if roi>=-margin else 0 for r,roi in zip(revs, roi_errors) ]
        avg_rev = np.mean(cond_revs)
        roi_sat = sum(roi_soft_sat) / numdays
        logger.info('accum: avg_rev={}; roi_sat={}'.format(avg_rev, roi_sat))
        logger.info('agent rev improvement: {:.2f}%'.format(
            (agent_rev/k2_rev - 1)*100))
        logger.info('-'*84)
        # write_values.extend(list(env_info))
        method_name = env_info[0]
        write_tb(env_info, keys, writer, prefix, method_name, logtype, step)

    return agent_metrics['accum_avg_norm_imp'], agent_metrics['normalized_imp_soft']

class LimitOpt(): # used for automated curriculum learning
    def __init__(self, agent, env, logger, writer):
        self.agent = agent
        self.env = env
        self.parameter = agent.agent.limit_parameter
        self.optimizer = agent.agent.limit_optimizer
        self.logger = logger
        self.writer = writer
        self.step = 0

    def init_limit_training(self):
        agent = self.agent

        self.optimizer.zero_grad()
        self.valid_day_cnt = 0
        self.limit_loss = torch.tensor(0.).to(device=agent.agent.get_device())

    def init_daywise(self):
        agent = self.agent
        self.diff_day_r = torch.tensor(0.).to(device=agent.agent.get_device())

    def update_and_compute_loss(self,):
        agent, env = self.agent, self.env
        diff_day_r = self.diff_day_r
        # global diff_day_r, limit_loss, valid_day_cnt
        # if agent.learn_limits:
        tmp = env.compute_reward_powerlaw_differentiable(None, None, agent.get_limit_parameter(detach=False))
        self.diff_day_r += tmp
        if env.past_day_ends(): #and agent.learn_limits:
            day_r = env.compute_reward_sparse(None, None)   # 0.5 because powerlaw use 0.5
            # print('day reward', day_r)
            if day_r > 0:
                day_r *= 0.5 * env.num_slot_in_day
                loss = (day_r - diff_day_r) ** 2
                # loss.backward()
                self.limit_loss += loss
                self.valid_day_cnt += 1
                # self.logger.info('valid day {}'.format(self.valid_day_cnt))

    def debug(self):
        agent = self.agent
        if (self.valid_day_cnt>0) and (self.valid_day_cnt) % 3 == 0:
            # last = agent.agent.limit_parameter.clone()
            # self.logger.info('last limit {}'.format(last.item()))
            # agent.agent.limit_parameter.grad /= valid_day_cnt
            self.limit_loss /= self.valid_day_cnt
            self.limit_loss.backward()
            self.optimizer.step()
            # if torch.abs(agent.agent.limit_parameter-last)<1e-4:
            #     print()
            self.logger.info('limit {} with loss {}'.format(agent.get_limit_parameter(), self.limit_loss))
            self.step += 1

            self.writer.add_scalar('{}/limit'.format('train_stats'), self.parameter.item(), self.step)
            self.writer.add_scalar('{}/limit_loss'.format('train_stats'), self.limit_loss.item(), self.step)
            self.writer.add_scalar('{}/sqrt_limit_loss'.format('train_stats'), np.sqrt(self.limit_loss.item()), self.step)

            self.init_limit_training()

def train_or_test_synchronous_exp(env, agent, logger, writer, window_size, exploration_num, C, istest, epoch_it=0, git=None, max_perf=-np.inf, need_exploration=True, is_cem=False, save_margin=0.1):
    istrain = not istest
    st = time.time()
    # do_incremental = not (istest and not agent.test_incremental)
    update_during_train = istrain
    is_train_env = not istest

    min_day = env.get_day()
    max_day = env.how_many_days()
    logger.info('traverse from {} to {} days'.format(min_day, max_day))

    day_cnt = 0+epoch_it*max_day
    state_preserver = env.state_preserver

    is_linear = agent._get_model_name()=='Linear' and 'yewu' in env.data_ver
    # limit_loss, valid_day_cnt = init_limit_training(agent)
    if agent.learn_limits:
        opt = LimitOpt(agent, env, logger, writer)
        opt.init_limit_training()

    # agent.set_period_exploration_factor(0.9)
    if not istest:
        agent.set_execution(False)
        while env.get_day()<max_day:
            # traverse through single day
            obs = env.get_obs(0., False, avg_value=is_linear)
            # diff_day_r = torch.tensor(0.).to(device=agent.agent.get_device())
            if agent.learn_limits: opt.init_daywise()
            while True:
                action = agent.batch_act(obs)
                ratio = agent.get_bid_ratio(state_preserver.last_ratio, action)
                # env step with the action and gives feedbacks
                po_rew, po_rew_normalized = env.slot_step_cem(action, ratio) if is_cem \
                    else env.slot_step(action, ratio, False, agent.get_multiplier(), agent.get_limit_parameter())  # this can be None

                if agent.learn_limits: opt.update_and_compute_loss()
                if po_rew is not None and env.use_history:
                    state_preserver.add_to_history((obs, action,
                                                    po_rew_normalized))  # 因为next obs是下一个obs，所以必须在next前面就加上history; 因为empty slot导致reward nan，所以要判断

                next_obs = env.get_obs(ratio, empty=env.past_day_ends(), avg_value=is_linear)
                po_rew_normalized = agent.postprocess_reward(obs, ratio, next_obs, po_rew_normalized)
                env.step_logging(True, state_preserver, po_rew, po_rew_normalized, ratio, agent.output_dir,
                                 agent._get_model_name())

                state_preserver._reset_slot()
                # env gives next obs, pause to save the whole transition
                agent.update_buffer((obs, action, 0 if po_rew is None else po_rew_normalized, env.past_day_ends(), next_obs))

                obs = next_obs
                if env.get_day() > 0 or is_cem:
                    agent.learn(can_train=update_during_train)

                if env.test_period_ends():
                    env.update_each_period()  # update recent ratios and loose-win states

                if env.past_day_ends(): break

            if agent.learn_limits: opt.debug()

            env.dayover_stats_update()
            env.dayover_logging(False, state_preserver, agent.output_dir)
            agent.dayover_logging()
            state_preserver._reset_day()
            log_as_final = istest and day_cnt==env.how_many_days()-1

            window_perf,perf = report('test' if istest else 'train', env, agent, agent.output_dir, writer=writer, logger=logger,
                   base_day=day_cnt, logtypes_elim=None,
                   window_size=window_size, #window_size,
                   log_as_final=log_as_final,
                    max_perf=max_perf,
                    )

            logger.info('one day takes {}'.format(time.time() - st))
            logger.info('+' * 84)
            # agent._reset_day()
            st = time.time()
            day_cnt += 1

            if env.past_week_ends():
                env.weekover_logging(False, state_preserver)
        logger.info('max day {} reached'.format(max_day))
        logger.info('average value per slot, {}'.format(state_preserver.avg_value_sum / (max_day*env.num_slot_in_day)))

    logger.info('start val epoch')
    agent.eval()
    env.set_step(0)
    env.reset_traversal()
    agent.set_execution(True)
    while env.get_day()<max_day:
        # traverse through single day
        obs = env.get_obs(0., False, avg_value=is_linear)

        if agent.learn_limits: opt.init_daywise()
        while True:
            action = agent.batch_act(obs)
            ratio = agent.get_bid_ratio(state_preserver.last_ratio, action)
            # env step with the action and gives feedbacks
            po_rew, po_rew_normalized = env.slot_step_cem(action, ratio) if is_cem \
                else env.slot_step(action, ratio, True, agent.get_multiplier(), agent.get_limit_parameter())  # this can be None
            if agent.learn_limits: opt.update_and_compute_loss()
            if po_rew is not None and env.use_history:
                state_preserver.add_to_history((obs, action,
                                                # env.get_obs(ratio, empty=env.past_day_ends(), force_no_history=True,
                                                #             avg_value=is_linear),
                                                po_rew_normalized))  # 因为next obs是下一个obs，所以必须在next前面就加上history; 因为empty slot导致reward nan，所以要判断

            next_obs = env.get_obs(ratio, empty=env.past_day_ends(), avg_value=is_linear)
            po_rew_normalized = agent.postprocess_reward(obs, ratio, next_obs, po_rew_normalized)
            env.step_logging(False, state_preserver, po_rew, po_rew_normalized, ratio, agent.output_dir,
                             agent._get_model_name())

            state_preserver._reset_slot()
            # env gives next obs, pause to save the whole transition
            agent.update_buffer((obs, action, 0 if po_rew is None else po_rew_normalized, env.past_day_ends(), next_obs))

            obs = next_obs

            if env.test_period_ends():
                env.update_each_period()  # update recent ratios and loose-win states

            if env.past_day_ends(): break

        if agent.learn_limits: opt.debug()
        env.dayover_stats_update()
        env.dayover_logging(True, state_preserver, agent.output_dir)
        agent.dayover_logging()
        state_preserver._reset_day()
        log_as_final = istest and day_cnt==env.how_many_days()-1

        window_perf,perf = report('eval' if not istest else 'test', env, agent, agent.output_dir, writer=writer, logger=logger,
               base_day=day_cnt, logtypes_elim=None,
               window_size=max_day,
               log_as_final=log_as_final,
                max_perf=max_perf,
                )

        logger.info('one day takes {}'.format(time.time() - st))
        logger.info('+' * 84)
        # agent._reset_day()
        st = time.time()
        day_cnt += 1

        if env.past_week_ends():
            env.weekover_logging(False, state_preserver)

    if istrain and window_perf > max_perf - 0.1:
        scaler = state_preserver.get_reward_std()
        agent.save(0, scaler)
        env.save(agent.output_dir)
        if window_perf > max_perf:
            max_perf = window_perf
        logger.info('agent has been saved as 0 with {}({}), scaler {}'.format(perf, window_perf, scaler))
    else:
        logger.info(
            'agent score {}({} in window {})'.format(perf, window_perf, max_day))

    env.set_step(0)
    env.reset_traversal()
    agent.set_train()

    return max_perf

def prepare_slot_generation(cfg): # prepare for slot-wise synthetic data
    pre_run_slot_generator(cfg, 'train', do_plot=False)
    pre_run_slot_generator(cfg, 'test')

def main(args, make_agent, train_before_test):

    test = False
    cfg, envname, exploration_num, num_slot, is_syn, debug, iid, loading, data_ver, subfolder, seed = prepare_and_seed(
        args)
    torch.manual_seed(seed)
    if args.prepare:
        return prepare_slot_generation(cfg)

    agent = make_agent(cfg, test, envname)

    reward_scaler = None
    writer = agent.writer
    logger = agent.logger

    prefix = '../data/'
    if loading == 'all':
        train_folder = prefix + subfolder + '_train_all'
        test_folder = prefix + subfolder + '_test_all'
    else:
        train_folder = test_folder = prefix + subfolder + '_each'
    test_only = args.test_only
    resume = args.resume
    if not is_syn:
        mapping = dict(v2='v6', v3='v7', v4='v9',
                       v3ood='v7ood', v3bood='v7ood', v3bLood='v7ood',
                       v3b='v7', v3bL='v7',
                       v4ood='v9ood', v4bood='v9ood', v4bLood='v9ood',
                       v4b='v9', v4bL='v9',
                       )
        # yewu_ver = mapping.get(data_ver[5:], 'v5')
        # yewu_v4xx ood/''
        if 'v4' in data_ver:
            yewu_ver = 'v9'
            if 'ood' in data_ver: yewu_ver += 'ood'
        elif 'v3' in data_ver:
            yewu_ver = 'v1'
            if 'ood' in data_ver: yewu_ver += 'ood'
        else: yewu_ver = mapping.get(data_ver[5:], 'v5')
        print('yewu ver, {}'.format(yewu_ver))
        ###############
        if int(yewu_ver[1]) in [1,9]:
            train_folder = test_folder = prefix + 'yewu-data/' + yewu_ver  # v9 or v9ood
            files = [prefix + 'yewu-data/all0221/' + x for x in os.listdir(train_folder)]
        #############################
        elif 9> int(yewu_ver[1]) >= 7: # deprecated
            train_folder = test_folder = prefix + 'yewu-data/all'
            # meta = pkl.load(open(train_folder + '/data_meta.pkl', 'rb'))
            meta = pkl.load(open('../data/yewu-data/all0221/data_meta_all_f.pkl', 'rb'))
            valid_days = meta['dynamics']['corresponding_days']
            day2idx = {day: idx for idx, day in enumerate(valid_days)}
            months = ['08', '09', '11', '12', '01', '02']
            day_by_month = {k: [] for k in months}
            for day in valid_days:
                fname = day.split('.')[-2]
                date = fname.split(os.path.sep)[-1]
                k = date[:2]
                day_by_month[k].append(day)
            month_set = ['08', '09', '11', '12'] if 'ood' not in yewu_ver else ['01','02']
            files = list(itertools.chain(*[sorted(day_by_month[k]) for k in month_set ]))
        else: # deprecated
            train_folder = test_folder = prefix + 'yewu-data/' + yewu_ver
            # meta = pkl.load(open(train_folder + '/data_meta.pkl', 'rb'))
            meta = pkl.load(open('../data/yewu-data/all0221/data_meta_all_f.pkl', 'rb'))
            files = list(meta['each_count'].keys())

        train_days = eval(cfg['data']['train_days'])
        random.seed(0)
        np.random.seed(0)
        if 'ood' in yewu_ver:
            cfg['data']['train_files'] = str([])
            cfg['data']['test_files'] = str(files[:train_days])
        else:
            print('choosing {} out of {}'.format(train_days, len(files)))

            train_files = sorted(random.sample(files, train_days))
            test_files = [x for x in files if x not in set(train_files)]
            # random.seed(seed)
            # train_files = files[:train_days]
            cfg['data']['train_files'] = str(train_files)
            cfg['data']['test_files'] = str(test_files)

    train_env = None
    if not test_only:
        train_env = gym.make(envname + '_{}-v0'.format('train'),
                             cfg=cfg, data_folder=train_folder)  # 'YewuSimulator-v0'
        train_env.seed(seed)
    test_env = gym.make(envname + '_{}-v0'.format('test'),
                        cfg=cfg, data_folder=test_folder)
    test_env.seed(seed)

    ################
    preserver = test_env.state_preserver

    if (test_only or resume) and agent._get_model_name()=='BaseAgent':
        reward_scaler = agent.restore()
        logger.info('restore scaler {}'.format(reward_scaler))
        if train_env: train_env.set_reward_scaler(reward_scaler)

    #######
    if not test_only:
        train_env.set_logger(logger)
        train_env.set_action_space(agent._get_action_space())

    test_env.set_logger(logger)
    output_dir = agent.output_dir
    test_env.set_action_space(agent._get_action_space())

    C = eval(cfg['data']['C'])
    window_size = eval(cfg['data']['window_size'])
    train_max_days = eval(cfg['data']['train_max_days'])
    nepoch = eval(cfg['data']['nepoch'])
    gamma_nepoch = eval(cfg['data']['gamma_nepoch'])
    #### censored model
    if torch.cuda.is_available():
        what_device = 'cuda'
    else:
        what_device = 'cpu'
    device = torch.device(what_device)
    model = None
    # torch.autograd.set_detect_anomaly(True)

    if not eval(cfg['data']['debug']) and args.ratio == 0:
        if not test_only: agent.validate_action_space(train_env)
        agent.validate_action_space(test_env)

    # agent.set_day(day_num)
    test_day_unit = 1

    train_before_test(args.synchronous, test_env, train_env, agent, model, test_day_unit, train_max_days, exploration_num, writer,
                      logger, C, window_size, nepoch=nepoch, gamma_nepoch=gamma_nepoch, restore=test_only,
                      do_random=not args.no_randomize, init_b=args.powerlaw_b, save_margin=args.save_margin)

    logger.info('all done')

import argparse
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_syn', type=str, default='none', help='use synthetic cost')
    parser.add_argument('--wo_lost', action='store_true', help='without lost data')
    parser.add_argument('--inc', action='store_true', help='finetune during test')
    parser.add_argument('--feat_wkday', action='store_true', help='use wkday feature')
    parser.add_argument('--feat_costs', type=int, default=0, help='use costs feature') # 0,1,2
    parser.add_argument('--penalty', type=float, default=20)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--buffer', type=int, default=4)
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--reward', type=int, default=11)
    parser.add_argument('--iid', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--simp', action='store_true')
    parser.add_argument('--norm_mean', action='store_true')
    parser.add_argument('--train_niter', type=int, default=1)
    parser.add_argument('--folder', type=str, default='debug')
    parser.add_argument('--replay', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--cem_ratio', type=float, default=1.)
    parser.add_argument('--elite', type=float, default=0.2)
    parser.add_argument('--exp_percent', type=float, default=0.2)
    parser.add_argument('--bins', type=int, default=100)
    parser.add_argument('--cem_std', type=float, default=0.33)
    parser.add_argument('--slot', type=int, default=30)
    parser.add_argument('--period', type=int, default=4)
    parser.add_argument('--nepoch', type=int, default=15)
    parser.add_argument('--gamma_nepoch', type=int, default=1)
    parser.add_argument('--loading', type=str, default='slot') # or single
    parser.add_argument('--data_ver', type=str, default='syn_v3') # v3 v4 v3L, v4L v3b v4b
    parser.add_argument('--train_days', type=str, default='30')
    parser.add_argument('--test_days', type=str, default='30')
    parser.add_argument('--buffer_size', type=int, default=5000)
    parser.add_argument('--ablation', type=int, default=0)
    parser.add_argument('--rev_scaler', type=float, default=None)
    parser.add_argument('--discount', type=float, default=0.97) # or 0.95
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--num_sample_per_day', type=int, default=2000000)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--kp', type=float, default=2.)
    parser.add_argument('--ki', type=float, default=0.)
    parser.add_argument('--kd', type=float, default=0.)
    parser.add_argument('--pe', type=int, default=0)
    parser.add_argument('--drop', type=float, default=0.)
    parser.add_argument('--agent', type=str, default='sac') # sacmc, td3mc, ddpgmc for mcb
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--restore_dir', type=str, default='')
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--prepare', action='store_true')
    # parser.add_argument('--original_reward', action='store_true')
    parser.add_argument('--no_randomize', action='store_true')
    parser.add_argument('--fc1_hidden', type=int, default=30)
    parser.add_argument('--use_history', type=int, default=0) # 0,1 for false,true
    parser.add_argument('--history_size', type=int, default=10)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--history_emb_dim', type=int, default=10)
    parser.add_argument('--history_type', type=str, default='sar')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--max_loss', type=float, default=15.)
    parser.add_argument('--powerlaw_b', type=float, default=0.2) # or 0.1 then 0.2
    parser.add_argument('--save_margin', type=float, default=0.1)
    parser.add_argument('--use_decoder', type=int, default=0)
    parser.add_argument('--reconstr_num', type=int, default=1)
    parser.add_argument('--stop_q_bp', action='store_true')
    parser.add_argument('--C', type=str, default=None)
    parser.add_argument('--lin_ratio', type=str, default='None')
    parser.add_argument('--use_curiosity', action='store_true')
    parser.add_argument('--curiosity_factor', type=float, default=20.)
    parser.add_argument('--budget', type=float, default=0)
    parser.add_argument('--synchronous', type=int, default=0) # 0,1 for false true
    parser.add_argument('--action_hi', type=float, default=4.)
    parser.add_argument('--norm_reward', type=int, default=0) # 0,1 for false true
    parser.add_argument('--budget_braking', action='store_true')
    parser.add_argument('--buffer_old', action='store_true')
    parser.add_argument('--clip_norm', type=float, default=10.)
    parser.add_argument('--encoder_type', type=str, default='transformer') # gaussian, transformer, lstm
    parser.add_argument('--kl_factor', type=float, default=0.1)
    parser.add_argument('--bidirection', type=int, default=0)
    parser.add_argument('--bert_pooler', type=str, default='mean') # encoder need try, decoder default to mean
    parser.add_argument('--no_entropy', type=int, default=0) # default use entropy
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--tune_multiplier', type=int, default=0) # default learn
    parser.add_argument('--init_multiplier', type=float, default=1.7)  # default learn
    parser.add_argument('--learn_limits', type=int, default=0) # not learning limits 0

    # parser.add_argument('--bert_hidden', type=int, default=128)
    # parser.add_argument('--bert_nhead', type=int, default=8)
    args = parser.parse_args()
    return args

def update_configs(args):
    cfg = configparser.ConfigParser(allow_no_value=True)
    cfg.read('./config.cfg')
    cfg['bert']['bidirection'] = str(args.bidirection)
    cfg['bert']['bert_pooler'] = args.bert_pooler
    cfg['mcb']['no_entropy'] = str(args.no_entropy)
    if args.seed is not None:
        cfg['data']['seed'] = str(args.seed)
    else:
        seed = np.uint32(time.time()*1e6)
        cfg['data']['seed'] = str(seed)
    cfg['data']['synchronous'] = str(bool(args.synchronous))
    # if args.synchronous==1 and args.nepoch<=5:
    #     cfg['data']['nepoch'] = str(15)
    # else: cfg['data']['nepoch'] = str(args.nepoch)
    cfg['data']['nepoch'] = str(args.nepoch)
    cfg['data']['data_seeds'] = str(data_seeds)
    cfg['data']['num_sample_per_day'] = str(args.num_sample_per_day)
    cfg['data']['use_syn'] = args.use_syn
    cfg['data']['iid'] = str(args.iid)
    cfg['data']['train_niter'] = str(args.train_niter)
    cfg['data']['draw'] = str(args.draw)
    cfg['rl_agent']['kl_factor'] = str(args.kl_factor)
    cfg['rl_agent']['budget_braking'] = str(args.budget_braking)
    cfg['rl_agent']['simplified'] = str(args.simp)
    cfg['rl_agent']['reward_norm_with_mean'] = str(args.norm_mean)
    cfg['rl_agent']['ratio_update_type'] = '1' if args.agent=='mcddpg' else str(args.ratio)
    cfg['rl_agent']['agent_type'] = args.agent
    cfg['rl_agent']['soft_q_lr'] = str(args.lr)
    cfg['rl_agent']['train_freq'] = str(args.train_freq)
    cfg['rl_agent']['max_loss'] = str(args.max_loss)
    cfg['rl_agent']['stop_q_bp'] = str(args.stop_q_bp)
    cfg['rl_agent']['use_decoder'] = str(bool(args.use_decoder))
    cfg['rl_agent']['use_curiosity'] = str(args.use_curiosity)
    cfg['rl_agent']['curiosity_factor'] = str(args.curiosity_factor)
    cfg['rl_agent']['norm_reward'] = str(bool(args.norm_reward))
    cfg['rl_agent']['learn_limits'] = str(args.learn_limits)
    if args.ablation == 17: recon_num = 2
    elif args.ablation == 16: recon_num = 1
    else: recon_num = args.reconstr_num
    cfg['rl_agent']['reconstruction_num'] = str(recon_num)
    cfg['rl_agent']['buffer_old'] = str(args.buffer_old)
    cfg['rl_agent']['clip_norm'] = str(args.clip_norm)
    cfg['rl_agent']['encoder_type'] = str(args.encoder_type)
    cfg['rl_agent']['tune_multiplier'] = str(bool(args.tune_multiplier))
    cfg['rl_agent']['init_multiplier'] = str(args.init_multiplier)
    cfg['cem']['elite_percent'] = str(args.elite)
    cfg['cem']['exp_percent'] = str(args.exp_percent)
    cfg['cem']['exp_nsample'] = str(args.bins)
    cfg['cem']['init_ratio_std'] = str(args.cem_std)
    cfg['cem']['init_ratio'] = str(args.cem_ratio)
    cfg['linear']['init_ratio'] = args.lin_ratio
    cfg['data']['slot_len'] = str(args.slot)
    cfg['data']['period_len'] = str(args.period)

    cfg['data']['gamma_nepoch'] = str(args.gamma_nepoch) if args.force else '2'
    cfg['data']['loading'] = 'each' if 'yewu' in args.data_ver else 'slot' # str(args.loading)
    # v5: perfect utility C=1; v6: noisy utility C=1; v7: noisy utility C=1.1
    # if args.C is not None: cfg['data']['C'] = str(args.C)
    # else: cfg['data']['C'] = str(1.1 if args.data_ver in ['syn_v4','syn_v7','yewu_v4'] else 1.0) #str(args.C)
    cfg['data']['C'] = str(1.0)
    cfg['rl_agent']['buffer_size'] = str(args.buffer_size)
    cfg['rl_agent']['ablation'] = str(args.ablation)
    # cfg['rl_agent']['norm_reward'] = str(not args.original_reward)
    cfg['data']['randomize_days'] = str(not args.no_randomize)
    cfg['rl_agent']['fc1_hidden'] = str(args.fc1_hidden)
    cfg['rl_agent']['use_history'] = str(bool(args.use_history))
    cfg['rl_agent']['history_size'] = str(args.history_size)
    cfg['rl_agent']['history_emb_dim'] = str(args.history_emb_dim)
    cfg['rl_agent']['history_type'] = str(args.history_type)
    cfg['rl_agent']['powerlaw_b'] = str(args.powerlaw_b)
    base_scaler = 1e-4
    # warning: rev_scaler 如果指定就不会做下面的scaling
    base_num = 2000000
    if args.rev_scaler is not None:
        base_scaler *= args.rev_scaler
    args.rev_scaler = base_scaler /(args.num_sample_per_day / base_num)
    print('rev scaler modified,', args.rev_scaler)
    cfg['data']['rev_scaler'] = str(args.rev_scaler)
    cfg['rl_agent']['discount'] = str(args.discount)
    cfg['pid']['kp'] = str(args.kp)
    cfg['pid']['ki'] = str(args.ki)
    cfg['pid']['kd'] = str(args.kd)
    cfg['pid']['ratio'] = str(args.ratio)
    cfg['rl_agent']['position_encoding'] = str(args.pe)
    cfg['data']['drop'] = str(args.drop)
    cfg['rl_agent']['use_bn'] = str(args.use_bn)
    if args.tau: cfg['rl_agent']['tau'] = str(args.tau)
    if args.train_days:
        cfg['data']['train_days'] = '30' if args.data_ver.startswith('yewu_v4') else args.train_days
    if args.test_days:
        cfg['data']['test_days'] = '30' if args.data_ver.startswith('yewu_v4') else args.test_days
    data_ver = args.data_ver
    data_folder = '{}_s{}'.format(data_ver, args.slot)
    envname = data_ver.split('_')[0]
    cfg['data']['envname'] = 'SynSimulator' if envname=='syn' else 'YewuSimulator'
    cfg['data']['data_folder'] = data_folder
    cfg['data']['version'] = data_ver
    if 'b' in data_ver:
        cfg['data']['budget'] = '0.8'
    else:
        cfg['data']['budget'] = '0' # str(args.has_budget)
    if args.restore_dir:
        cfg['rl_agent']['restore_dir'] = args.restore_dir
    if args.wo_lost:
        cfg['data']['include_loosing'] = 'False'
    if args.inc:
        cfg['data']['test_inc'] = 'True'
    if args.feat_wkday:
        cfg['rl_agent']['wkday_feat'] = 'True'
    cfg['rl_agent']['future_costs'] = str(args.feat_costs)
    cfg['hypers']['penalty'] = str(args.penalty)
    if args.debug:
        cfg['data']['debug'] = 'True'
        cfg['data']['train_days'] = '5'
        cfg['data']['test_days'] = '5'
    if args.folder != '':
        def_folder = cfg['data']['output_dir']
        folder = os.path.join(def_folder, args.folder)
        cfg['data']['output_dir'] = folder
    if args.replay == 0:
        cfg['rl_agent']['replay_scheme'] = 'uniform'
    else:
        cfg['rl_agent']['replay_scheme'] = 'prioritized'

    cfg['rl_agent']['gamma'] = str(args.gamma)

    cfg['rl_agent']['buffertype'] = str(args.buffer)
    cfg['rl_agent']['clearbuffer'] = str(args.clear)
    cfg['rl_agent']['rewardtype'] = str(args.reward)
    cfg['rl_agent']['resume'] = str(args.resume)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if torch.cuda.is_available():
        print('using gpu {}'.format(args.gpu))

    return cfg

def prepare_and_seed(args):
    cfg = update_configs(args)
    envname = cfg['data']['envname']  # 'YewuSimulator'#'AuctionEmulator-v0'
    exploration_num = eval(cfg['data']['exp_num'])
    num_slot = eval(cfg['data']['slot_len'])
    is_syn = 'syn' in envname.lower()
    debug = args.debug
    iid = args.iid
    loading = args.loading
    data_ver = args.data_ver
    names = data_ver.split('_')
    if iid: names.insert(1, 'iid')
    subfolder = '_'.join(names)

    seed = eval(cfg['data']['seed'])
    # random.seed(seed)
    # np.random.seed(seed)

    return cfg, envname, exploration_num, num_slot, is_syn, debug, iid, loading, data_ver, subfolder, seed

def sync_weights(local_model, target_model, tau, syn_method='avg'):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    if syn_method=='avg':
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    else:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)


