import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# from pytorch_lightning.tuner import random_search
from ads.models.Model import TabularDataset, AutoencoderModel
import sys

sys.path.insert(0, '../../2022_Haghighi_NatureMethods/utils/')
print(sys.path)
from readProfiles import *
from pred_models import *
from data_utils import load_data


def train_autoencoder(data_path, batch_size, latent_size, lr, max_epochs, val_check_interval, l2_lambda):
    data = pd.read_csv(data_path)
    train_data, val_data = train_test_split(data, test_size=0.2)
    train_dataset = TabularDataset(train_data)
    val_dataset = TabularDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = AutoencoderModel(input_size=data.shape[1], latent_size=latent_size, lr=lr, l2_lambda=l2_lambda)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best_autoencoder",
        save_top_k=1,
        mode="min",
    )

    logger = TensorBoardLogger("tb_logs", name="autoencoder")

    trainer = Trainer(
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        gpus=torch.cuda.device_count(),
        precision=16,
        deterministic=True,
        benchmark=True,
        auto_lr_find=True
    )

    # result = trainer.tuner.random_search(
    #     model,
    #     train_loader,
    #     val_loader,
    #     num_trials=10,
    #     timeout=600,
    #     objective="val_loss",
    #     direction="minimize",
    # )
    result = trainer.fit(model, train_loader, val_loader)

    return result.best_model, result.best_hyperparameters


if __name__ == "__main__":


    procProf_dir = '/sise/assafzar-group/assafzar/genesAndMorph'
    # dataset type: CDRP, CDRP-bio, LINCS, LUAD, TAORF
    dataset = 'CDRP-bio'

    # CP Profile Type options: 'augmented' , 'normalized', 'normalized_variable_selected'
    profileType = 'normalized_variable_selected'

    ################################################
    # filtering to compounds which have high replicates for both GE and CP datasets
    # highRepOverlapEnabled=0
    # 'highRepUnion','highRepOverlap'
    # filter_perts = ''
    # repCorrFilePath = '../results/RepCor/RepCorrDF.xlsx'
    #
    # filter_repCorr_params = [filter_perts, repCorrFilePath]
    #
    # ################################################
    # pertColName = 'PERT'
    #
    # if dataset == 'TAORF':
    #     filter_perts = ''
    # else:
    #     filter_perts = 'highRepOverlap'
    #
    # if filter_perts:
    #     f = 'filt'
    # else:
    #     f = ''

    seed_everything(42)

    batch_size = 64
    latent_size = 16
    lr = 0.001
    max_epochs = 100
    val_check_interval = 0.5
    l2_lambda = 0.01


    cp, cp_features = load_data(procProf_dir, dataset, profileType)


    # def train_autoencoder(data_path, batch_size, latent_size, lr, max_epochs, val_check_interval, l2_lambda):
    # data = pd.read_csv(data_path)

    # train_data, val_data = train_test_split(cp_scaled, test_size=0.2)
    # train_dataset = TabularDataset(train_data[cp_features])
    # val_dataset = TabularDataset(val_data[cp_features])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = AutoencoderModel(input_size=len(cp_features), latent_size=latent_size, lr=lr, l2_lambda=l2_lambda)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best_autoencoder",
        save_top_k=1,
        mode="min",
    )

    logger = TensorBoardLogger("tb_logs", name="autoencoder")

    trainer = Trainer(
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=max_epochs,
        val_check_interval=val_check_interval,
        gpus=torch.cuda.device_count(),
        precision=16,
        deterministic=True,
        benchmark=True,
        auto_lr_find=True
    )

    trainer.fit(model, train_loader, val_loader)
    # trainer.fit()
    # result = trainer.tuner.random_search(
    #     model,
    #     train_loader,
    #     val_loader,
    #     num_trials=10,
    #     timeout=600,
    #     objective="val_loss",
    #     direction="minimize",
    # )
    trainer.test(dataloaders=[val_loader,test_loader_mocks,test_loader_treated])
    # return result.best_model, result.best_hyperparameters