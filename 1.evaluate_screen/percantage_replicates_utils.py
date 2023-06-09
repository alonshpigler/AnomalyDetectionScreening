import random
from collections import defaultdict
from statistics import median
import numpy as np
from tqdm import tqdm


def get_median_score(cpds_list, df):
    """
    This function calculates the median score for each compound based on its replicates
    """

    cpds_median_score = {}
    for cpd in cpds_list:
        cpd_replicates = df[df.index.isin([cpd], 2)].copy()
        median_val = get_median_correlation(cpd_replicates)

        cpds_median_score[cpd] = median_val

    return cpds_median_score


def get_median_correlation(cpd_replicates):
    cor_mat = cpd_replicates.astype('float64').T.corr(method='pearson').values
    cor_mat = np.nan_to_num(cor_mat)

    if len(cor_mat) == 1:
        median_val = 1
    else:
        median_val = np.median(cor_mat[np.triu_indices(len(cor_mat), k=1)])
    return median_val


def get_replicates_dict(cpds_list, df):
    ordered_replicates = defaultdict()
    for cpd in cpds_list:
        cpd_replicates = df[df.index.isin([cpd], 2)].copy()

        ordered_replicates[cpd] = cpd_replicates

    return ordered_replicates


def drop_cpds_with_null(df):
    """
    This function drop compounds with median scores of 1
    or null values in any of the dose points (1-6)
    """
    cpds_with_null = []
    for cpd in df.index:
        if any(df.loc[cpd] == 1) | any(df.loc[cpd].isnull()):
            cpds_with_null.append(cpd)
    df.drop(cpds_with_null, axis=0, inplace=True)

    return df


def get_duplicate_replicates(cpd_df, min_num_reps=4):
    compounds_cnt = cpd_df.index.get_level_values(2).value_counts()
    dup_compounds = (compounds_cnt[compounds_cnt >= min_num_reps]).index
    replicates_df = cpd_df[cpd_df.index.isin(dup_compounds, 2)]
    cpds = replicates_df.index.get_level_values(2).unique()

    return replicates_df, cpds


def get_replicates_score(cpd_df, fields=None):
    if fields is not None:
        fields = cpd_df.columns

    replicates_df, cpds = get_duplicate_replicates(cpd_df)
    cpds_score_df = get_median_score(cpds, replicates_df[fields])

    return cpds_score_df


def get_null_distribution_replicates(
        all_replicates,
        cpds,
        rand_num=1000,
        num_reps_in_rand=4
):
    """
    This function returns a null distribution dictionary, with no_of_replicates(replicate class)
    as the keys and 1000 lists of randomly selected replicate combinations as the values
    for each no_of_replicates class per DOSE(1-6)
    """
    random.seed(1903)

    null_distribution_reps = {}

    for replicate in tqdm(cpds):
        replicate_list = []
        for idx in range(rand_num):
            start_again = True
            while (start_again):
                rand_cpds = get_random_replicates(
                    all_replicates.index.get_level_values(2),
                    num_reps_in_rand,
                    replicate
                )

                if rand_cpds not in replicate_list:
                    start_again = False

            replicate_list.append(rand_cpds)

        null_distribution_reps[replicate] = replicate_list

    return null_distribution_reps


def get_random_replicates(all_replicates, no_of_replicates, replicate):
    """
    This function return a list of random replicates that are not of the same compounds
    or found in the current cpd's size list
    """
    while (True):
        random_replicates = random.sample(set(all_replicates), no_of_replicates)
        # random_replicates = list(all_replicates.sample(no_of_replicates).index.get_level_values(2))
        # print(random_replicates)
        if not (any(replicate in rep for rep in random_replicates)):
            # & (check_similar_replicates(random_replicates, dose, cpd_replicate_dict))):
            break

    return random_replicates


def assert_null_distribution(null_distribution_reps):
    """
    This function assert that each of the list in the 1000 lists of random replicate 
    combination for each no_of_replicate class are distinct with no duplicates
    """
    duplicates_reps = {}
    for keys in null_distribution_reps:
        null_dist = null_distribution_reps[keys]
        for reps in null_dist:
            dup_reps = []
            new_list = list(filter(lambda x: x != reps, null_dist))
            if (len(new_list) != len(null_dist) - 1):
                dup_reps.append(reps)
        if dup_reps:
            if keys not in duplicates_reps:
                duplicates_reps[keys] = [dup_reps]
            else:
                duplicates_reps[keys] += [dup_reps]
    return duplicates_reps


def calc_null_dist_median_scores(df, replicate_lists):
    """
    This function calculate the median of the correlation 
    values for each list in the 1000 lists of random replicate 
    combination for each no_of_replicate class
    """
    # df = df.set_index('replicate_name').rename_axis(None, axis=0)
    # df.drop(['Metadata_broad_sample', 'Metadata_pert_id', 'Metadata_dose_recode', 
    #          'Metadata_Plate', 'Metadata_Well', 'Metadata_broad_id', 'Metadata_moa', 
    #          'broad_id', 'pert_iname', 'moa'], 
    #          axis = 1, inplace = True)

    median_corr_list = []
    for rep_list in replicate_lists:
        # df_reps = df.loc[rep_list].copy()
        # df_reps = df[df.index.isin(rep_list)]
        df_reps = df[df.index.get_level_values(2).isin(rep_list)]
        reps_corr = df_reps.astype('float64').T.corr(method='pearson').values
        median_corr_val = median(list(reps_corr[np.triu_indices(len(reps_corr), k=1)]))
        median_corr_list.append(median_corr_val)
    return median_corr_list


def get_null_dist_median_scores(null_distribution_cpds, df, features = None):
    """ 
    This function calculate the median correlation scores for all 
    1000 lists of randomly combined compounds for each no_of_replicate class 
    """
    
    null_distribution_medians = {}
    
    if features is not None:
        df = df.loc[:,features]
        
    for key in tqdm(null_distribution_cpds):
        # print(key)

        replicate_median_scores = calc_null_dist_median_scores(df, null_distribution_cpds[key])
        null_distribution_medians[key] = replicate_median_scores
    return null_distribution_medians


def get_p_value(median_scores_list, df, cpd_name, method):
    # def get_p_value(median_scores_list, df, dose_name, cpd_name):
    """
    This function calculate the p-value from the 
    null_distribution median scores for each compound
    """
    # actual_med = df.loc[cpd_name, dose_name]
    actual_med = df.loc[cpd_name][method]
    p_value = np.sum(median_scores_list >= actual_med) / len(median_scores_list)
    return p_value


def get_moa_p_vals(null_dist_median, df_med_values, method='map'):
    """
    This function returns a dict, with compounds as the keys and the compound's 
    p-values as the values
    """
    null_p_vals = {}
    # for key in null_dist_median:
    # print(key)
    # df_replicate_class = df_med_values[df_med_values['no_of_replicates'] == key]
    # TODO - to adjust to no_of_replicates
    df_replicate_class = df_med_values.copy()
    
    for cpd in df_replicate_class.index:
        # dose_p_values = []
        # for num in dose_list:
        # dose_name = 'dose_' + str(num)
        # cpd_p_value = get_p_value(null_dist_median[key], df_replicate_class, cpd,method)

        cpd_p_value = get_p_value(null_dist_median[cpd], df_replicate_class, cpd, method)
        # dose_p_values.append(cpd_p_value)
        null_p_vals[cpd] = cpd_p_value
        # df_replicate_class.loc[cpd][f'{method}_p_val'] = cpd_p_value
    sorted_null_p_vals = {key: value for key, value in sorted(null_p_vals.items(), key=lambda item: item[0])}
    return sorted_null_p_vals
    # return df_replicate_class

    

if __name__ == '__main__':

    null_dist_path, slice_size, slice_id, dest = sys.argv[1:]
    slice_size = int(slice_size)
    slice_id = int(slice_id)
    binarize = False
    t = 2


    zscores = {
        # '4to1':{'path':f'/storage/users/g-and-n/tabular_models_results/41/ALL/zscores'},
        # '2to1':{'path':f'/storage/users/g-and-n/tabular_models_results/21/ALL/zscores'},
        # '5to5': {'path': f'/storage/users/g-and-n/tabular_models_results/111/ALL/zscores'},
        # 'raw': {'path': f'/storage/users/g-and-n/tabular_models_results/30000/results/z_scores/pure/raw'},
        # '5to5':{'path':f'/sise/assafzar-group/g-and-n/tabular_models_results/4000/zscores'},
        # 'raw':{'path': f'/sise/assafzar-group/g-and-n/plates/raw_normalized'}
        # '5to5':{'path':f'/storage/users/g-and-n/tabular_models_results/55/ALL/zscores'},
    }
    exps = [5001,5002,5003,5004,5005,5006,5007,5008,5009,5010,5011,5012]
    for exp in exps:
        exp_name = str(exp)
        zscores[exp_name] = {}
        zscores[exp_name]['path']= f'/sise/assafzar-group/g-and-n/tabular_models_results/{exp}/zscores'
    # get plate numbers
    cur_fld = f'/storage/users/g-and-n/tabular_models_results/41/ALL/'
    all_plates = glob(os.path.join(cur_fld, 'zscores', '*'))
    plate_nums = [os.path.split(plate)[-1] for plate in all_plates]

    # load zscores
    for model in zscores.keys():
        plates = [os.path.join(zscores[model]['path'], plate) for plate in plate_nums]
        zscores[model]['all'] = pd.concat([pd.read_csv(pth, index_col=[0, 1, 2, 3]) for pth in plates])

        # Binarize
        if binarize:
            if model == 'raw':
                zscores[model]['all'] = zscores[model]['all'].abs()
            zscores[model]['all'] = zscores[model]['all'].apply(lambda s: s.apply(lambda x: 1.0 if x >= t else 0.0))

    # Drop columns that are not in other results
    # find joint columns
    joint_cols = []
    for model in zscores.keys():
        model_cols = zscores[model]['all'].columns
        if len(joint_cols) == 0:
            joint_cols = model_cols
        else:
            joint_cols = np.intersect1d(joint_cols, model_cols)

    # drop non-joint columns
    for model in zscores.keys():
        zscores[model]['all'].drop(columns=[c for c in zscores[model]['all'].columns if c not in joint_cols],
                                   inplace=True)
        print(zscores[model]['all'].shape)

    for model in zscores.keys():
        zscores[model]['trt'] = zscores[model]['all'].query('Metadata_ASSAY_WELL_ROLE == "treated"')
        # zscores[model]['ctl'] = zscores[model]['all'].query('Metadata_ASSAY_WELL_ROLE == "mock"')


    with open(null_dist_path, 'rb') as f:
        # Load the pickle file
        null_distribution_replicates = pickle.load(f)


    cpds = list(null_distribution_replicates.keys())
    cpds.sort()
    cpds = cpds[slice_id * slice_size:(slice_id + 1) * slice_size]

    for model in zscores.keys():
        print(5, model)
        cur_dest = os.path.join(dest, model)
        os.makedirs(cur_dest, exist_ok=True)
        cur_dest = os.path.join(cur_dest, f'{slice_id}.pickle')
        res = {}

        for cpd in cpds:
            print(cpd)
            cpd_replicates = zscores[model]['trt'][zscores[model]['trt'].index.isin([cpd], 2)].copy()
            cpd_med_score = get_median_correlation(cpd_replicates)
            del cpd_replicates

            null_dist = null_distribution_replicates[cpd]
            null_dist_scores = []
            for null_idxs in null_dist:
                cur_null = zscores[model]['trt'][zscores[model]['trt'].index.isin(null_idxs)].copy()
                curr_null_score = get_median_correlation(cur_null)
                null_dist_scores.append(curr_null_score)
                del cur_null

            res[cpd] = (cpd_med_score, null_dist_scores)

        with open(cur_dest, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
