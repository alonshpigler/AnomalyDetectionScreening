import os

# from readProfiles import read_replicate_level_profiles
import numpy as np
import pandas as pd
from pycytominer import feature_select
from pycytominer.operations import variance_threshold, get_na_columns
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from dataset_paper_repo.utils.readProfiles import read_replicate_level_profiles

#TODO: transform each dataset info to dicts
ds_info_dict={'CDRP':['CDRP-BBBC047-Bray',['Metadata_Sample_Dose','pert_sample_dose'],['Metadata_ASSAY_WELL_ROLE','mock'],'Metadata_Plate'],
              'CDRP-bio':['CDRPBIO-BBBC036-Bray',['Metadata_Sample_Dose','pert_sample_dose'],['Metadata_ASSAY_WELL_ROLE','mock'],'Metadata_Plate'],
              'TAORF':['TA-ORF-BBBC037-Rohban',['Metadata_broad_sample','pert_id',]],
              'LUAD':['LUAD-BBBC041-Caicedo',['x_mutation_status','allele']],
              'LINCS':['LINCS-Pilot1',['Metadata_pert_id_dose','pert_id_dose'],['Metadata_pert_type','control'],'Metadata_plate_map_name']}


# index_fields =


class TabularDataset(Dataset):
  def __init__(self, data):
    self.data = data.to_numpy().astype(np.float32)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


def load_data(procProf_dir, dataset, profileType, plate_normalized=0):
  [cp_data_repLevel, cp_features], [l1k_data_repLevel, l1k_features] = \
    read_replicate_level_profiles(procProf_dir, dataset, profileType, plate_normalized)

  # l1k = l1k_data_repLevel[[pertColName] + l1k_features]
  # cp = cp_data_repLevel[[pertColName] + cp_features]
  #
  # if dataset == 'LINCS':
  #   cp['Compounds'] = cp['PERT'].str[0:13]
  #   l1k['Compounds'] = l1k['PERT'].str[0:13]
  # else:
  #   cp['Compounds'] = cp['PERT']
  #   l1k['Compounds'] = l1k['PERT']

  return cp_data_repLevel, cp_features


def prepare_data(data, config, features, modality ='CellPainting', do_fs = True):

  data_path = f'{config["data_dir"]}/preprocessed_data/{ds_info_dict[config["dataset"]][0]}/{modality}/replicate_level_cp_{config["profile_type"]}.csv.gz'
  if 'fs' in config["profile_type"] and not os.path.exists(data_path):
    # do feature selection - only only on mock data!!
    cp_features, cols_to_drop = feature_selection(data[data[ds_info_dict[config['dataset']][2][0]] == ds_info_dict[config['dataset']][2][1]], features)
    data = data.drop(cols_to_drop, axis=1)
    data.loc[:,cp_features] = data.loc[:,cp_features].interpolate()
    data.to_csv(data_path, compression='gzip')
    config['overwrite_data_creation'] = True

  else:
    # cp_features=None
    cp_features = data.columns[data.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
    # cp_features = np.intersect1d(cp_features, data.columns)

  print(f'number of features for training is {len(cp_features)}')

  # devide to train, val, test_mocks, and test_treated
  datasets = pre_process(data, config, cp_features,overwrite=config['overwrite_data_creation'])

  # construct dataset
  dataset_modules = {}
  for key in datasets.keys():
    dataset_modules[key] = TabularDataset(datasets[key][cp_features])

  # construct dataloaders
  dataloaders = {}
  for key in datasets.keys():
    if key == 'train':
      #TODO: consider moving dataloaders to model to enable auto-tune of batch size
      dataloaders[key] = DataLoader(dataset_modules[key], config['batch_size'], shuffle=True)
    else:
      dataloaders[key] = DataLoader(dataset_modules[key], config['batch_size'])

  return datasets, dataloaders, cp_features


def pre_process(data, config, features = None,overwrite = False):
  #TODO: debug why not working with LINCS - nan in data (see Zernike_3_3)
  # take control data for training

  data_path = os.path.join(config['data_dir'], 'anomaly_output', config['dataset'], config['profile_type'])
  train_path = os.path.join(data_path, f'input_data_train.csv')

  if not os.path.exists(train_path) or overwrite:
    control_data = data[data[ds_info_dict[config['dataset']][2][0]] == ds_info_dict[config['dataset']][2][1]]

    # split data with equal samples from different plates (ds_info_dict[config['dataset'][3]])
    splitter = GroupShuffleSplit(test_size=config['test_split_ratio'], n_splits=2, random_state=7)
    split = splitter.split(control_data, groups=control_data[ds_info_dict[config['dataset']][3]])
    train_inds, test_inds = next(split)

    train_data_all = control_data.iloc[train_inds]
    test_data_mocks = control_data.iloc[test_inds]

    # train_data, val_data = train_test_split(mock_data, test_size=0.4)
    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7)
    split = splitter.split(train_data_all, groups=train_data_all[ds_info_dict[config['dataset']][3]])
    train_inds, val_inds = next(split)

    train_data = train_data_all.iloc[train_inds]
    val_data = train_data_all.iloc[val_inds]

    # leave out control data
    test_data_treated = data[data[ds_info_dict[config['dataset']][2][0]] != ds_info_dict[config['dataset']][2][1]]

    datasets_pre_normalization = {
      'train': train_data,
      'val': val_data,
      'test_ctrl': test_data_mocks,
      'test_treat': test_data_treated
    }

    for set in datasets_pre_normalization.keys():
      datasets_pre_normalization[set].to_csv(os.path.join(data_path, f'input_data_{set}.csv'),compression='gzip')

    datasets = datasets_pre_normalization.copy()

    print('normalizing to training set...')

    # scale to training set
    scaler_cp = preprocessing.StandardScaler()
    datasets['train'].loc[:, features] = scaler_cp.fit_transform(datasets_pre_normalization['train'][features].values.astype('float64'))
    datasets['train'].loc[:, features] = datasets['train'].loc[:, features].fillna(0)
    for set in list(datasets.keys())[1:]:
      datasets[set].loc[:, features] = scaler_cp.transform(datasets[set][features].values.astype('float64'))
      datasets[set].loc[:, features] = datasets[set].loc[:, features].fillna(0)

    for set in datasets.keys():
      datasets[set].to_csv(os.path.join(data_path, f'input_data_{set}_zscores_by_train.csv'),compression='gzip')


  else:
    datasets_pre_normalization={}
    datasets = {}
    sets = ['train','val','test_ctrl','test_treat']
    for s in sets:
      datasets_pre_normalization[s] = pd.read_csv(os.path.join(data_path, f'input_data_{s}.csv'),compression='gzip')
      datasets[s] = pd.read_csv(os.path.join(data_path, f'input_data_{s}_zscores_by_train.csv'),compression='gzip')


  raw_path = os.path.join(os.path.join(data_path, f'input_data_test_treated_zscores_by_test.csv'))
  if not os.path.exists(raw_path) or overwrite:

    print('calclating raw measurements...')

    test_data_mocks_n_by_test = datasets_pre_normalization['test_ctrl'].copy()
    test_data_treated_n_by_test = datasets_pre_normalization['test_treat'].copy()

    # raw_data_calculation
    scaler_cp = preprocessing.StandardScaler()
    test_data_mocks_n_by_test.loc[:, features] = scaler_cp.fit_transform(datasets_pre_normalization['test_ctrl'][features].values.astype('float64'))
    test_data_mocks_n_by_test.loc[:, features]= test_data_mocks_n_by_test.loc[:, features].fillna(0)
    test_data_treated_n_by_test.loc[:, features] = scaler_cp.transform(datasets_pre_normalization['test_treat'][features].values.astype('float64'))
    test_data_treated_n_by_test.loc[:, features] = test_data_treated_n_by_test.loc[:, features].fillna(0)


    test_data_mocks_n_by_test.to_csv(os.path.join(data_path, f'input_data_test_ctrl_zscores_normalized_by_test.csv'),compression='gzip')
    test_data_treated_n_by_test.to_csv(os.path.join(data_path, f'input_data_test_treated_zscores_by_test.csv'),compression='gzip')
  else:
    print('skipping raw calculation')

  return datasets


def scale_data_by_set(scale_by, sets, features):

  # scaler_ge = preprocessing.StandardScaler()
  scaler_cp = preprocessing.StandardScaler()
  # l1k_scaled = l1k.copy()
  # l1k_scaled[l1k_features] = scaler_ge.fit_transform(l1k[l1k_features].values)
  # cp_scaled = cp.copy()
  scale_by[features] = scaler_cp.fit_transform(scale_by[features].values.astype('float64')).fillna(0)

  scaled_sets = []
  scaled_sets.append(scale_by)
  for set in sets:
    scaled_set = set.copy()
    scaled_set[features] = scaler_cp.transform(set.values.astype('float64')).fillna(0)
    scaled_sets.append(scaled_set)

  return scaled_sets


def feature_selection(data, features=None):
    """
    Perform feature selection by dropping columns with null or
    only zeros values, and highly correlated values from the data.

    params:
    dataset_link: string of github link to the consensus dataset

    Returns:
    data: returned consensus dataframe

    """
    null_vals_ratio = 0.05
    thrsh_std = 0.001

    print(f'number of features before feature selection: {len(features)}')

    # cols2remove_manyNulls = [i for i in features if
    #                          (data[i].isnull().sum(axis=0) / data.shape[0]) \
    #                          > null_vals_ratio]
    # cols2remove_lowVars = data[features].std()[
    #   data[features].std() < thrsh_std].index.tolist()
    #
    cols2remove_manyNulls = get_na_columns(
      population_df=data,
      features=features,
      samples="all",
      cutoff=null_vals_ratio,
    )
    cols2remove_lowVars = variance_threshold(
      population_df=data,
      features=features,
      samples="all",
      freq_cut=0.05,
      unique_cut=0.01,
    )

    cols2removeCP = cols2remove_manyNulls + cols2remove_lowVars

    cp_features = list(set(features) - set(cols2removeCP))
    print(f'number of features after removing cols with nulls and low var: {len(cp_features)}')

    cp_data_repLevel = data.drop(cols2removeCP, axis=1)
    cols2remove_highCorr = get_highly_correlated_features(cp_data_repLevel, cp_features)

    cp_features = list(set(cp_features) - set(cols2remove_highCorr))
    print(f'number of features after removing high correlated features: {len(cp_features)}')
    cp_data_repLevel = data.drop(cols2remove_highCorr, axis=1)
    # cp_data_repLevel[cp_features] = cp_data_repLevel[cp_features].interpolate()

    cols2removeCP += cols2remove_highCorr
    return cp_features, cols2removeCP


def get_highly_correlated_features(data, features,threshold=0.95):

    # Compute the correlation matrix
    corr_matrix = data[features].corr(method='pearson').abs()

    # get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    return to_drop
