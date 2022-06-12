import sys

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from synthesize import find_oracle
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
from glob import glob
from collections import OrderedDict
from scipy import sparse
from pulp import * #
import itertools
from collections import defaultdict

folder = '../data/yewu-data/invalid'
meta_name = '/data_meta_invalid'
save_dynamics = False
os.makedirs(folder, exist_ok=True)
np.random.seed(0)


def get_selected(train_days, sample_num=400000):
    selected_idxes = []
    for day in train_days:
        tmp = pd.read_csv(day, header=None, usecols=[16])
        num = tmp.notnull().sum()
        print(day, num)
        idxes = np.arange(len(tmp))[tmp.notnull().values.reshape(-1)]
        selected = np.random.choice(idxes, sample_num)
        print(len(selected))
        selected_idxes.append(selected)
    return selected_idxes


def merge_hours(folder):
    print('processing folder {}'.format(folder))
    files = glob(folder + '/*')
    if len(files) == 0:
        print('folder {} is empty'.format(folder))
    dfs = []
    for file in tqdm(files):
        ret = get_df_from_file(file)
        if ret is None: continue
        dfs.append(ret)
    return None if len(dfs)==0 else pd.concat(dfs)


def get_df_from_file(fp):
    try:
        tmpdf = pd.read_csv(fp, header=None)
    except Exception as e:
        print('reading {} with exception {}, skip.'.format(fp, str(e)))
        return None

    cols1 = ['bidid', 'time', 'wkday', 'hr', 'min', 'user', 'pid', 'all', 'k2', 'dsp_pctr',
            'pppc',  # 10
            'sessid', 'creativeid', 'pctr', 'upbidprice', 'exptime', 'costprice',
            'click', 'clk2',
            'rev']

    cols2 = ['bidid', 'time', 'wkday', 'hr', 'min', 'user', 'pid', 'k2', 'dsp_pctr',
             'pppc',  # 10
             'sessid', 'creativeid', 'pctr', 'upbidprice', 'costprice',
             'click', 'clk2',
             'rev']

    if len(tmpdf.columns)<len(cols1):
        cols = cols2
    else: cols = cols1

    tmpdf.columns = cols
    notnull_selector = tmpdf['costprice'] > 0  # cost not none is won data
    # won_selector = tmp[8] * tmp[9] * tmp[10] * 1000 > tmp[16]
    # all_idxes = np.arange(len(tmp))
    # notnull_idxes = all_idxes[notnull_selector]
    # won_idxes = all_idxes[np.logical_and(won_selector, notnull_selector)]
    tmpdf = tmpdf[notnull_selector]
    # dayid = tmpdf['wkday'].iloc[0]
    # filter = tmpdf.apply(lambda row: row[10] > 0 and row[13] > 0 and row[2] == dayid, axis=1)
    # tmpdf = tmpdf[filter]  # pppc



    final = tmpdf[['wkday', 'hr', 'min', 'user', 'pid', 'k2', 'dsp_pctr',
                   'pppc',  # 10
                   'pctr', 'costprice',
                   'click',
                   'rev',
                   ]].copy()
    del tmpdf
    return final


def get_df_single_day(path, sample_num=None):
    if os.path.isdir(path):
        df = merge_hours(path)
    else:
        df = get_df_from_file(path)

    if df is not None:
        print('{} has {} available bids'.format(path, df.shape[0]))
        df['rev'] = df.rev.fillna(0.)
        return df
    else: return None
    # if sample_num > len(won_idxes):
    #     won_num = len(won_idxes)
    # else:
    #     ratio = len(won_idxes) / len(notnull_idxes)
    #     won_num = int(ratio * sample_num)
    #     won_idxes = np.random.choice(won_idxes, size=won_num, replace=False)
    # loose_num = sample_num - won_num
    #
    #
    # loose_pool_selector = np.logical_and(~won_selector, notnull_selector)
    # loose_idxes = np.random.choice(all_idxes[loose_pool_selector], size=loose_num, replace=False)
    #
    # print('won', len(won_idxes), 'loose', len(loose_idxes))
    # # all_won_idxes.append(won_idxes)
    # # all_loose_idxes.append()
    #
    # won_loose_idxes = won_idxes.tolist() + loose_idxes.tolist()
    # tmpdf = tmp.iloc[won_loose_idxes]
    # tmpdf = tmpdf[tmpdf[2] == dayid]
    # tmpdf['loose'] = np.ones(tmpdf.shape[0])
    # tmpdf.iloc[:won_num, -1] = 0


def dump_df_each(train_days, sample_num=None, slot=30, plot=False, budget_suffix='', L_suffix=''):
    """
    budget: default 0, no budget
    C: default 0.9, so that budgets is less and roi can be higher
    """
    global folder
    """
       load from folder+'data_meta.pkl'
       budgets save in meta['budgets']
       policies save in meta['policies'][<csvpath>][<policy>][<metrics>]
       dump to folder+'/data_meta_{suffix}.pkl'
    """
    # L suffix: 'f' for fixed L, 'L' for diverse L, 'L2' for CPC
    # budget suffix: 'b' for diverse budget.
    # '' fixed L; 'L' diverse L depends on ''; 'b' depends on ''; 'bL' depends on 'b'; 'bL2' depends on 'b'

    # suffixes
    # if budget > 0:
    #     budget_suffix = '_b{}'.format(budget)
    # else:
    #     budget_suffix = ''
    # if diverse_constraints:
    #     L_suffix = '_L'
    # else:
    #     L_suffix = ''
    if budget_suffix.startswith('_b'): budget = eval(budget_suffix[2:])
    else: budget = 0
    if 'L' in L_suffix: diverse_constraints = True
    else:
        diverse_constraints = False
        init_C = 0
        # if 'f' in L_suffix:
        #     init_C = eval(L_suffix.split('f')[-1])
        #     print('using init_L {}'.format(init_C))


    already_dumped_files = set()
    num_slot_in_day = 24 * 60 // slot
    # oracles = dict()
    oracle_revs = dict()
    # slot_oracles = dict()
    count = OrderedDict()
    min_ratio, max_ratio = np.inf, 0
    pids_unique = set()
    users_unique = set()
    aders_unique = set()
    all_costs = []
    cost_hist = []
    rois = []
    included_files = []
    # budgets = dict()
    # Ls = dict()
    # determine the diverse constraints (budgets or Ls)
    # 'f' fixed L; 'L' diverse L depends on 'bf'; 'bf' depends on 'f'; 'bL' depends on 'bf'; 'bL2' depends on 'bf'

    ## diverse budgets.
    if budget>0: # b depends on ''; bL bL2 depends on b
        meta_path = folder + '/{}{}.pkl'.format(meta_name, '_f0.95') # to obtain budget, all bx uses the cost of f
        print('loading {}, to obtain the budget'.format(meta_path))
        assert os.path.exists(meta_path), 'previous meta {} does not exist.'
        old_meta = pkl.load(open(meta_path, 'rb'))
        old_policies = old_meta['policies']
        budgets = {csvpath: old_policies[csvpath]['slot_oracle']['cost'] * budget for csvpath in old_policies}
        # if not diverse_constraints: # exactly 'b' mode, requires to construct budgets
        #
        # else:
        #     assert 'budgets' in old_meta
        #     budgets = old_meta['budgets']

    if diverse_constraints: # diverse L will depend on bf (only f makes the constraint 1.0)
        meta_path = folder + '{}{}.pkl'.format(meta_name, '_b0.7_f1.0')
        assert os.path.exists(meta_path), 'previous meta {} does not exist.'.format(meta_path)
        print('loading {}, to obtain the diverse L'.format(meta_path))
        old_meta = pkl.load(open(meta_path, 'rb'))
        old_policies = old_meta['policies']

        Ls = dict()
        if L_suffix.endswith('L'):  # not cpc constraint
            for csvpath in old_meta['each_count']:
                roi = old_policies[csvpath]['slot_oracle']['rev'] / (old_policies[csvpath]['oracle']['cost'] + 1e-4)
                rd = np.random.randint(-10,1) / 100.
                L = max(np.floor(roi * 100) / 100. + rd, 1.) # at least 1.0
                Ls[csvpath] = L
        else:  # cpc constraint
            raise NotImplementedError()
    ####################### done loading constraints

    all_dynamics = dict()
    invalid_days = dict()
    policies = defaultdict(dict) # dict of setting to dict; dict of day to dict; dict of two policies to dict
    budgets = defaultdict(dict)
    Ls = defaultdict(dict)
    oracles = defaultdict(dict)
    for idx, day in enumerate(train_days): #enumerate each csv
        ## prepare data
        csv_path = folder + '/{}.csv'.format(day.split(os.path.sep)[-1])
        print('='*20)
        print(csv_path)
        print('=' * 20)
        # policies[csv_path] = dict()
        # determine constraints
        if diverse_constraints:
            if csv_path not in Ls: continue
            C = Ls[csv_path]
        else:
            C = init_C
        if day in already_dumped_files:
            continue
        # read data
        if os.path.exists(csv_path):
            already_dump = True
            print('{} already exists.'.format(csv_path))
            final = pd.read_csv(csv_path)
        else:
            final = get_df_single_day(day)  # all data by default
            if final is None:
                print('skip {}'.format(day))
                continue
            print('dumping {}'.format(day))
            final.to_csv(csv_path, index=False)
        ################### done loading data

        ## stats with the data
        count[csv_path] = final.shape[0]
        included_files.append(day)
        pids_unique.update(set(final.pid.unique().tolist()))

        # find global oracles
        values = final.pctr * final.pppc
        print('find oracles')
        exact_day = day.split(os.path.sep)[-1]
        threshes = np.arange(0.005, 2., 0.005) if exact_day.startswith('01') or exact_day.startswith('02') else np.arange(0.01, 4., 0.01)
        oracle_revcost_by_slot_thrsh = []

        cols = final.columns
        final.loc[:, 'time_slot'] = final.hr * 60 // slot + final['min'] // slot
        final.loc[:, 'value'] = values

        sorev, socost, sowin = 0, 0, 0
        total_bids = 0
        slot_dynamics = dict(ctr=dict(), cvr=dict(), cost_rev_joint=dict())
        # init slot dynamics, including these data statistics
        keys = ['costprice', 'rev', 'value', 'pppc']
        for key in keys:
            slot_dynamics[key+'_mean'] = dict()
            slot_dynamics[key + '_mode'] = dict()
            # slot_dynamics[key + '_hist'] = dict()

        valid_slot = 0
        # find slot oracle
        for k, group in final.groupby('time_slot'):
            evalue = group.value.values
            bids = np.outer(threshes, evalue)  # m,n
            costs = group.costprice.values * 1e-3
            revs = group.rev.values
            wins = (bids > costs[None, :])
            winsum = wins.sum(1)
            won_rsum = np.einsum('mn,n->m', wins, revs)
            won_csum = np.einsum('mn,n->m', wins, costs)
            tmp = np.stack([won_rsum, won_csum, winsum], axis=-1) # thrsh,3

            oracle_revcost_by_slot_thrsh.append(tmp)  # slot,thrsh,3
            # won_csum, winnum = tmp[:, 1], tmp[:, 2]
            nbid = evalue.shape[0]
            total_bids += nbid

            # # compute stats only at the beginning
            if L_suffix=='_f' and save_dynamics:
                boolarr = revs > 0
                slot_dynamics['cost_rev_joint'][k] = np.histogram2d(costs[boolarr]*1e3, revs[boolarr], bins=(500, 500))
                click_count = nbid- group['click'].isna().sum()
                conversion_count = boolarr.sum()
                slot_dynamics['cvr'][k] = conversion_count/click_count
                # record dynamics:
                for key in keys:
                    tmp = group[key][group[key]>0]
                    mean = tmp.mean()
                    slot_dynamics[key+'_mean'][k] = mean
                    # slot_dynamics[key + '_hist'][k] = np.histogram(tmp, bins=100)
                    if sum(tmp <= mean) == 0:
                        mode = 0
                    else: mode = tmp[tmp<=mean].mode().values[0]

                    # if (mode<10 and key=='cost'):
                    #     import ipdb; ipdb.set_trace()
                    slot_dynamics[key + '_mode'][k] = mode
                slot_dynamics['ctr'][k] = click_count / nbid

            if group.shape[0]>1000: valid_slot += 1
        #################### done slot oracle

        # oracle of whole day
        slot_thrsh_rcw = np.stack(oracle_revcost_by_slot_thrsh, axis=1)  # thrsh,slot, 3
        rcw_cum = slot_thrsh_rcw.cumsum(1) # thrsh, slot, 3
        rcw_thrsh = slot_thrsh_rcw.sum(1)  # thrsh, 3

        roi_cum = rcw_cum[...,0] / (rcw_cum[...,1] + 1e-4)
        roi_thrsh = rcw_thrsh[:, 0] / (rcw_thrsh[:, 1] + 1e-4)


        print('num bid {}'.format(total_bids))

        # linear program for slot oracle
        numslot_exact = slot_thrsh_rcw.shape[1] #
        x_selection = [['s{}_th{}'.format(t, idx) for idx, th in enumerate(threshes)] for t in range(numslot_exact)]
        flattened_x = list(itertools.chain(*x_selection))
        item_costs = dict(
            ('s{}_th{}'.format(slot, idx), slot_thrsh_rcw[idx, slot, 1]) for slot in range(numslot_exact) for idx, th
            in enumerate(threshes))
        item_revs = dict(
            ('s{}_th{}'.format(slot, idx), slot_thrsh_rcw[idx, slot, 0]) for slot in range(numslot_exact) for
            idx, th in enumerate(threshes))
        x_vars = LpVariable.dicts('item', flattened_x, 0, 1, LpBinary)

        rsum = lpSum([item_revs[i] * x_vars[i] for i in flattened_x])
        csum = lpSum([item_costs[i] * x_vars[i] for i in flattened_x])
        obj = rsum

        ############ lp definition done
        # f: fixed L constraint, C=prescribed value, current budget=0
        # bf: budget is restricted, fixed L constraint may be loose. C=prescribed value, current budget = 0.7x f's cost
        # bL: budget tight, L loose and diverse. C=bf's roi+rand, current budget = 0.7x f's cost
        # bL1: budget tight, L tight and diverse. C=bf's roi, current budget = 0.7x f's cost
        # bL2: both budget and constraint might be tight. C=bf's cpc, current budget = 0.7x f's cost
        # L: diverse L constraint, C=bf's roi+rand, current budget=0

        for setting in ['f','b', 'bf','bL','bL1','L']:
            if setting=='f':
                current_budget = 0
                current_L = 1.0
            elif setting=='b':
                prev_policy = policies['f'][csv_path]['slot_oracle']
                current_budget = 0.8 * prev_policy['cost']
                current_L = 0
            elif 'b' in setting:
                prev_policy = policies['f'][csv_path]['slot_oracle']
                current_budget = 0.8 * prev_policy['cost']
                prev_policy = policies['b'][csv_path]['slot_oracle']
                prev_roi = np.floor(prev_policy['rev']/(prev_policy['cost']+1e-4)*100)
                
                if setting=='bf': current_L = 1.0
                elif setting=='bL':
                    current_L = max(prev_roi+np.random.randint(-10,1), 100) / 100.
                elif setting=='bL1':
                    current_L = max(prev_roi, 100) / 100
                else: raise NotImplementedError()
            else: # L
                prev_policy = policies['b'][csv_path]['slot_oracle']
                prev_roi = np.floor(prev_policy['rev']/(prev_policy['cost']+1e-4)*100)
                current_L = max(prev_roi+np.random.randint(-10,1), 100) / 100.
                current_budget = 0
            budgets[setting][csv_path] = current_budget
            Ls[setting][csv_path] = current_L

            slot_oracles, sorev, socost, sowin, optimal_lp = solve_lp(x_vars, threshes, numslot_exact, obj, rsum, csum, x_selection,
                                        current_L, current_budget, slot_thrsh_rcw)

            ans_thresh, rev, cost, win = solve_ratio(current_budget, current_L,
                                                     roi_cum, rcw_cum, threshes, roi_thrsh, rcw_thrsh)

            print('='*20)
            print('{}: current L {}, current budget {}'.format(setting, current_L, current_budget))
            policies[setting][csv_path] = dict(slot_oracle=dict(rev=sorev, cost=socost, nwin=sowin, ratios=slot_oracles),
                                               oracle=dict(rev=rev, cost=cost, nwin=win, ratio=ans_thresh))
            oracles[setting][csv_path] = ans_thresh

            for k,d in policies[setting][csv_path].items():
                if k=='nbid': continue
                r,c,w = [d[x] for x in ['rev','cost','nwin']]
                print('policy {}: roi={},rev={},cost={}, wr={}'.format(k, r/(c+1e-4), r, c, w/total_bids))
            print('oracle vs slot oracle: {}'.format(round((rev/(sorev+1e-4)-1)*100,2)))
            # print('k2 vs slot oracle: {}'.format(round((k2_rev/(sorev+1e-4)-1)*100,2)))
        ################ done computing oracle
        k2_bool = final.k2 * final.pppc * final.pctr > final.costprice * 1e-3
        k2_cost = final.costprice[k2_bool].sum() * 1e-3
        k2_rev = final.rev[k2_bool].sum()
        k2_win = k2_bool.sum()
        policies['online'][csv_path] = dict(nbid=total_bids,
                                            k2=dict(rev=k2_rev, cost=k2_cost, nwin=k2_win))

        all_dynamics[csv_path] = slot_dynamics

        if valid_slot<num_slot_in_day or (rev / (cost+1e-4) <C and not optimal_lp): # check if both solution is incorrect
            invalid_days[csv_path] = valid_slot

        del final

        # final.loc[:, 'oracle'] = thresh

    all_days = set(all_dynamics.keys())
    broken_days = set(invalid_days.keys())
    valid_days = list(all_days-broken_days)
    broken_days = list(broken_days)
    dynamics_to_save = dict()
    spmat = None
    if save_dynamics: # don't save in budget setting
        for key in keys:
            for suffix in ['_mean','_mode']:
                    dynamics_to_save[key+suffix] = np.asarray([[all_dynamics[day][key+suffix][i] for i in range(num_slot_in_day)] for day in valid_days], dtype=np.float32)
        dynamics_to_save['ctr'] = np.asarray([[all_dynamics[day]['ctr'][i] for i in range(num_slot_in_day)] for day in valid_days], dtype=np.float32)
        dynamics_to_save['cvr'] = np.asarray(
            [[all_dynamics[day]['cvr'][i] for i in range(num_slot_in_day)] for day in valid_days], dtype=np.float32)
        # dynamics_to_save['cost_rev_hist'] = np.stack([[all_dynamics[day]['cost_rev_joint'][i][0] for i in range(num_slot_in_day)] for day in valid_days]).astype(np.int32)
        spmat = sparse.vstack([sparse.csr_matrix(all_dynamics[day]['cost_rev_joint'][i][0].astype(np.int32)) for day in valid_days for i in range(num_slot_in_day)])
        # spmat = sparse.csr_matrix(dense)
        dynamics_to_save['cost_rev_xbin'] = np.stack(
            [[all_dynamics[day]['cost_rev_joint'][i][1] for i in range(num_slot_in_day)] for day in valid_days]).astype(
            np.float32)
        dynamics_to_save['cost_rev_ybin'] = np.stack(
            [[all_dynamics[day]['cost_rev_joint'][i][2] for i in range(num_slot_in_day)] for day in valid_days]).astype(
            np.float32)

    dynamics_to_save['corresponding_days'] = valid_days

    min_ratio = min([min(d.values()) for setting, d in oracles.items()])
    max_ratio = max([max(d.values()) for setting, d in oracles.items()])
    total_count = sum(count.values())
    meta = dict(count=total_count,
                each_count=count,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                pids=list(pids_unique),
                users=list(users_unique),
                aders=list(aders_unique),
                # oracles=oracles,  # dict, csv_path to ratio
                # oracle_revs=oracle_revs,
                dynamics=dynamics_to_save,
                invalid=invalid_days,
                # slot_oracles=slot_oracles,
                policies=policies,
                budgets=budgets,
                Ls=Ls
                # costs=all_costs,
                # cost_hist=cost_hist,
                # rois=rois
                # files=included_files
                # slot_oracles=slot_oracles
                )
    print('data total bids {}, min ratio {}, max ratio {}'.format(total_count, min_ratio, max_ratio))

    if save_dynamics:
        sparse.save_npz(folder + '/cost_rev_hist2.npz', spmat)
    pkl.dump(meta, open(folder+'/{}{}{}.pkl'.format(meta_name, budget_suffix, L_suffix), 'wb'))

def solve_lp(x_vars, threshes, num_slot_in_day, obj, rsum, csum, x_selection, C, current_budget, slot_thrsh_rcw):
    prob = LpProblem('slot_oracle', LpMaximize)
    prob += obj
    # constraints
    ## L constraints
    if C!=0:  # include 'f', 'L', 'L2'
        # ROI>L => rsum/csum > L => rsum >= Lcsum
        Lconstraint = rsum - C * csum >= 0
        prob += Lconstraint
    ## budget constraint
    if current_budget > 0:
        prob += csum <= current_budget
    ## each group must have one active
    for xvar_each_slot in x_selection:
        prob += lpSum(x_vars[i] for i in xvar_each_slot) == 1
    ################### constraints done
    prob.solve()
    # print(LpStatus[prob.status])
    if LpSolution[prob.status]==LpSolutionOptimal:
        optimal = True
    else: optimal = False

    # return [[value(x_vars['s{}_th{}'.format(0, idx)]) for idx in range(len(threshes))] for t in range(num_slot_in_day)]
    slot_oracles = dict()
    sorev, socost, sowin = 0, 0, 0
    for t in range(num_slot_in_day):
        # aaa = [value(x_vars['s{}_th{}'.format(0, idx)]) for idx in range(len(threshes))]
        # assert sum(aaa)==1, 'each group should sum to 1'
        for idx, th in enumerate(threshes):
            val = value(x_vars['s{}_th{}'.format(t, idx)])
            if val == 1:
                slot_oracles[t] = th
                srev, scost, swin = slot_thrsh_rcw[idx, t]
                sorev += srev
                socost += scost
                sowin += swin
                break

    return slot_oracles, sorev, socost, sowin, optimal

def solve_ratio(current_budget, C, roi_cum, rcw_cum, threshes, roi_thrsh, rcw_thrsh):
    if current_budget > 0:  # diverse constraints
        # current_budget = budgets[csv_path]
        # bool_arr = np.logical_and(rcw_thrsh[:, 1] <= current_budget , bool_arr)
        bool_arr = (roi_cum > C) & (rcw_cum[..., 1] <= current_budget)  # thrsh, slot
        print('=' * 20)
        print('current budget {}'.format(current_budget))
        thrsh_cumslot_rev = rcw_cum[..., 0]  # thrsh, slot
        numslot = thrsh_cumslot_rev.shape[1]
        # import ipdb; ipdb.set_trace()
        if bool_arr.sum() != 0:
            min_idx_flattened = thrsh_cumslot_rev[bool_arr].argmax()  #
            ans_thresh = np.tile(threshes.reshape((-1, 1)), (1, numslot))[bool_arr][min_idx_flattened]  # flattened
            # col_idx = min_idx_flattened % numslot
            rev, cost, win = rcw_cum[bool_arr][min_idx_flattened]
        else:
            ans_thresh, rev, cost, win = 0, 0, 0, 0
            print('no global oracle')
    else:  # no budget
        bool_arr = roi_thrsh > C
        if bool_arr.sum() != 0:
            indices = np.arange(len(roi_thrsh))
            oracle_rev_thr = rcw_thrsh[:, 0]
            min_idx = oracle_rev_thr[bool_arr].argmax()
            ans_idx = indices[bool_arr][min_idx]
            ans_thresh = threshes[bool_arr][min_idx]
            rev, cost, win = rcw_thrsh[ans_idx]
        else:
            bool_arr = (roi_cum > C)  # thrsh, slot
            if bool_arr.sum() != 0:
                thrsh_cumslot_rev = rcw_cum[..., 0]  # thrsh, slot
                min_idx_flattened = thrsh_cumslot_rev[bool_arr].argmax()  #
                # row_idx = min_idx_flattened // numslot
                # import ipdb; ipdb.set_trace()
                ans_thresh = np.tile(threshes.reshape((-1, 1)), (1, numslot))[bool_arr][
                    min_idx_flattened]  # flattened
                # col_idx = min_idx_flattened % numslot
                rev, cost, win = rcw_cum[bool_arr][min_idx_flattened]
            else:
                ans_thresh, rev, cost, win = 0, 0, 0, 0
                print('no global oracle')

    return ans_thresh, rev, cost, win

# wkday_range = [x%7 for x in range(2,9)]
def get_all_dfs(train_days, selected_idxes, won_num):
    dfs = []
    for dayid, day, idxes in tqdm(zip(wkday_range, train_days, selected_idxes)):
        df = pd.read_csv(day, header=None)
        tmp = df.iloc[idxes]
        tmp = tmp[tmp[2] == dayid]
        tmp['loose'] = np.ones(tmp.shape[0]).astype(bool)
        tmp.loc[:won_num, 'loose'] = False
        dfs.append(tmp)

    return dfs


if __name__ == '__main__':
    # no budget: fixed L, diverse L according to optima of fixed L. suffix: '' for fixed L, 'L' for diverse L
    # diverse budget (0.8 xcost according to optima of fixed L): no L constraint (b), fixed L (bf), diverse L according to optima of no L.
    # suffix: 'b' for diverse budget.
    # diverse budget, CPC constraint, maximize profit. suffix: 'L2' for CPC.

    # L suffix: '' for fixed L, 'L' for diverse L, 'L2' for CPC
    # budget suffix: 'b' for diverse budget.
    # 'f' fixed L; 'L' diverse L depends on 'f'; 'bf' depends on 'f'; 'bL' depends on 'bf'; 'bL2' depends on 'bf'

    # srcfolder = '../rl_data/ood'
    # files = glob(srcfolder + '/*')
    args = sys.argv
    budget = eval(args[1]) # [0,1]
    budget_suffix = '_b{}'.format(budget) if budget>0 else ''
    L_suffix = args[2] # '' or _f or _L or _L2
    print('preparing budget setting {}'.format(budget))
    files = [os.path.join(folder, file.split('.')[0]) for file in os.listdir(folder) if
             file.endswith('.csv') or
             (not file.startswith('.') and os.path.isdir(os.path.join(folder, file)))]
    print(files)
    # files = ['0901']
    # files = [files[-1], files[0]]
    slot = 30
    dump_df_each(files, slot=slot, plot=False, budget_suffix=budget_suffix, L_suffix=L_suffix)


