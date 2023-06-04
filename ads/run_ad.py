import numpy as np
import pandas as pd
from lightning.pytorch.tuner import Tuner
from matplotlib import pyplot as plt
from sklearn import preprocessing
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
# from pytorch_lightning.tuner import random_search
from ads.eval_utils import plot_latent_effect
from ads.models.AEModel import AutoencoderModel

# sys.path.insert(0, '../../2022_Haghighi_NatureMethods/utils/')
# print(sys.path)
# from dataset_paper_repo.utils.readProfiles import *
# import dataset_paper_repo.utils.readProfiles
# import dataset_paper_repo.utils.pred_models
# from dataset_paper_repo.utils.pred_models import *
from ads.models.CAE import ConcreteAutoencoderFeatureSelector
from data_utils import load_data, pre_process, prepare_data
import os
from typing import Dict, Union
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns

# from pytorch_lightning.utilities.cli import LightningCLI


def train_autoencoder(config: Dict[str, Union[str, float, int]],losses = {}) -> pl.LightningModule:
    # Read data
    cp, cp_features = load_data(config['data_dir'], config['dataset'], config['profile_type'])

    datasets, dataloaders,cp_features = prepare_data(cp, config, cp_features)

    # Initialize model
    hparams = {'input_size': len(cp_features),
        'latent_size': config['latent_dim'],
        'l2_lambda': config['l2_lambda'],
        'lr': config['lr'],
    }

    # Set up logger and checkpoint callbacks
    # logger = TensorBoardLogger(save_dir=config['tb_logs_dir'], name=config['exp_name'])
    logger = CSVLogger(save_dir=config['tb_logs_dir']),
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=30,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['ckpt_dir'],
        filename='autoencoder-{epoch:02d}-{val_loss:.2f}',
        save_top_k=config['save_top_k'],
        monitor='val_loss',
        mode='min'
    )
    progressbar_callback = TQDMProgressBar(refresh_rate=150)
    # Train model

    #TODO: move to lightning
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback,early_stop_callback,progressbar_callback],
        max_epochs=config['max_epochs'],
        accelerator="auto",
        # progress_bar_refresh_rate=50,
        precision=16 if config['use_16bit'] else 32
        # deterministic=True,
        # fast_dev_run=config['fast_dev_run']
    )

    # Run lr finder
    # lr_finder = trainer.tuner.lr_find(model, ...)

    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, dataloaders['train'],dataloaders['val'])

    # Inspect results
    # fig = lr_finder.plot()
    # fig.show()
    # suggested_lr = lr_finder.suggestion()
    # hparams['lr']=suggested_lr

    # batch_size_finder = tuner.scale_batch_size(model, dataloaders['train'],dataloaders['val'])
    # fig = batch_size_finder.plot()
    # fig.show()
    # suggested_bs = batch_size_finder.suggestion()
    # hparams['batch_size']=suggested_bs

    #TODO: add support to 'model' variable
    if config['model'] == 'ae':
        model = AutoencoderModel(hparams)

        trainer.fit(model, dataloaders['train'],dataloaders['val'])

    # NOT SUPPORTED YET
    elif config['model'] == 'cae':
        print('Concrete AEs not supported yet...')
        #TODO: when this works, run over K, do Yuval check of K hyperparameter

        # Initialize the feature selector model
        K = 10  # Number of features to select
        output_function = nn.Linear(K, len(cp_features))  # Update with your desired output function
        model = ConcreteAutoencoderFeatureSelector(K, output_function)

        #TODO: test if CAE fit works
        selector = ConcreteAutoencoderFeatureSelector(hparams)
        trainer.fit(model, dataloaders['train'],dataloaders['val'])

        #TODO: test if extraction of important features work
        selector.compute_probs(trainer.concrete_select)

        # Extract the indices of the most important features
        indices = selector.get_indices()

        # Extract the mask indicating the most important features
        mask = selector.get_mask()

        # Extract the most important features from the original data
        selected_features = mask[:, indices]
        #TODO: add support for test

    # Test model
    test_dataloaders = ['train','val','test_ctrl', 'test_treat']


    for data_subset in test_dataloaders:
        dataloader = dataloaders[data_subset]
        # dict_res = trainer.test(model, dataloader)
        # for res in dict_res:
        #     losses[data_subset][res].append(dict_res[res])
        losses[data_subset] = trainer.test(model, dataloader)

    # disable grads + batchnorm + dropout
    torch.set_grad_enabled(False)
    model.eval()
    all_preds = {}
    x_recon_preds = {}
    z_preds = {}
    predict_dataloaders = ['test_ctrl', 'test_treat']
    for subset in list(dataloaders.keys())[2:]:
        dataloader = dataloaders[subset]
        x_recon_preds[subset] =[]
        z_preds[subset] = []
        for batch_idx, batch in enumerate(dataloader):
            x_recon_pred, z_pred = model.predict_step(batch, batch_idx)
            x_recon_preds[subset].append(x_recon_pred.numpy())
            z_preds[subset].append(z_pred.numpy())


    x_recon_ctrl = np.concatenate(x_recon_preds['test_ctrl'])
    z_pred_ctrl = np.concatenate(z_preds['test_ctrl'])

    x_recon_treat = np.concatenate(x_recon_preds['test_treat'])
    z_pred_treat = np.concatenate(z_preds['test_treat'])

    # process and save anomaly detection output
    test_ctrl_out = datasets['test_ctrl'].copy()
    test_treat_out = datasets['test_treat'].copy()

    test_ctrl_out.loc[:,cp_features] = x_recon_ctrl
    test_treat_out.loc[:,cp_features] = x_recon_treat

    test_ctrl_out.to_csv(os.path.join(config['data_dir'], 'anomaly_output', config['dataset'],config['profile_type'],f'ad_out_ctrl.csv'),compression='gzip')
    test_treat_out.to_csv(os.path.join(config['data_dir'], 'anomaly_output', config['dataset'],config['profile_type'], f'ad_out_treated.csv'),compression='gzip')

    #TODO: add z_pred_saving and normalizing, keeping required indices.
    # z_pred_subset.to_csv(os.path.join(config['data_dir'], 'anomaly_output', config['dataset'],f'embedding_test_ctrl.csv'))

    test_ctrl_out_normalized = datasets['test_ctrl'].copy()
    test_treat_out_normalized = datasets['test_treat'].copy()

    scaler_cp = preprocessing.StandardScaler()
    test_ctrl_out_normalized.loc[:,cp_features] = scaler_cp.fit_transform(test_ctrl_out[cp_features].values.astype('float64'))
    test_treat_out_normalized.loc[:,cp_features] = scaler_cp.transform(test_treat_out[cp_features].values.astype('float64'))

    test_treat_out_normalized.to_csv(os.path.join(config['data_dir'], 'anomaly_output', config['dataset'],config['profile_type'], f'ad_out_ctrl_zscores.csv'),compression='gzip')
    test_treat_out_normalized.to_csv(
        os.path.join(config['data_dir'], 'anomaly_output', config['dataset'],config['profile_type'], f'ad_out_treated_zscores.csv'),compression='gzip')

    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    print(metrics.dropna(axis=1, how="all").head())
    sns.relplot(data=metrics, kind="line")
    plt.show()
    return model, losses


if __name__ == '__main__':

    seed_everything(42)
    import yaml

    # Read YAML file
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    print('start training...')

    os.makedirs(os.path.join(config['data_dir'], 'anomaly_output'), exist_ok=True)
    os.makedirs(os.path.join(config['data_dir'], 'anomaly_output',config['dataset']), exist_ok=True)
    os.makedirs(os.path.join(config['data_dir'], 'anomaly_output', config['dataset'], config['profile_type']), exist_ok=True)

    tune_ldims = False
    if tune_ldims:
        l_dims = [32]
        all_res = []
        pcc_res =[]
        for l_dim in l_dims:
            config['latent_size'] = l_dim
            model, res = train_autoencoder(config)
            all_res.append(res)
            pcc_res.append(res['val'][0]['pcc'])

        #TODO: test plot latent effect
        plot_latent_effect(pcc_res, l_dims)

        best_ldim_ind = np.argmax(all_res)

        config['latent_size'] = l_dims[best_ldim_ind]

    model, res = train_autoencoder(config)
    res


    # model