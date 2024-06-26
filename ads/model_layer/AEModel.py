import pandas as pd
import numpy as np
import torch
from pytorch_lightning.cli import ReduceLROnPlateau
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
# import lightning.pytorch as pl
from torchmetrics.functional import pearson_corrcoef,r2_score
# from pytorch_lightning.metrics.functional import mse


class Autoencoder(nn.Module):
  def __init__(self, input_size, latent_size, deep_decoder=False, encoder_type=False, dropout=0):
    super(Autoencoder, self).__init__()

    if encoder_type=='shallow':
      self.encoder = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        # nn.LeakyReLU(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, latent_size),
        # nn.ReLU(),
        # nn.LeakyReLU(0.1),
        # nn.Linear(32, latent_size)
      )
    elif encoder_type=='deep':
      self.encoder = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        # nn.LeakyReLU(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        # nn.LeakyReLU(0.1),
        nn.Linear(64, latent_size)
      )
    else:
      self.encoder = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        # nn.LeakyReLU(0.1),
        nn.Linear(256, 128),
        # nn.ReLU(),
        # nn.Linear(64, 32),
        nn.ReLU(),
        # nn.LeakyReLU(0.1),
        nn.Linear(128, latent_size)
      )
    if deep_decoder:
      self.decoder = nn.Sequential(
        # nn.Linear(latent_size, input_size),
        nn.ReLU(),
        nn.Linear(latent_size, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, input_size)
      )
    else:
      self.decoder = nn.Sequential(
        nn.Linear(latent_size, input_size)
      )
  def decode(self, z):
    return self.decoder(z)
  def encode(self, x):
    return self.encoder(x)
  
  def forward(self, x):
    z = self.encoder(x)
    x_recon = self.decoder(z)
    return x_recon,z


class AutoencoderModel(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.test_step_outputs = []

    self.save_hyperparameters(hparams)

    self.autoencoder = Autoencoder(
      input_size=self.hparams.input_size,
      latent_size=self.hparams.latent_size,
      deep_decoder=self.hparams.deep_decoder,
      encoder_type=self.hparams.encoder_type,
      dropout=self.hparams.dropout
    )

    self.l2_loss = nn.MSELoss(reduction='mean')
    self.l1_loss = nn.L1Loss(reduction='mean')


  # def forward(self, x):
  #   z = self.autoencoder.encode(x)
  #   x_recon = self.autoencoder.decode(z)
  #   # x_recon, z = self.autoencoder(x)
  #   return x_recon
  def predict(self, x):
    x_recon, z = self.autoencoder(x)
    return x_recon
  
  def get_weights(self, module ='encoder', layer_num = 1):
    if module == 'encoder':
      return self.autoencoder.encoder[layer_num-1].weight
    elif module == 'decoder':
      return self.autoencoder.decoder[layer_num-1].weight
  
  def update_weights(self,new_weights, module ='encoder', layer_num = 1):
    if module == 'encoder':
      self.autoencoder.encoder[layer_num-1].weight.data = new_weights
    elif module == 'decoder':
      self.autoencoder.decoder[layer_num-1].weight.data = new_weights

  def training_step(self, x):
    # x = batch
    x_recon, z = self.autoencoder(x)

    recon_loss = self.l2_loss(x_recon, x)
    l2_reg = self.hparams.l2_lambda * self.l2_loss(x_recon,torch.zeros_like(x_recon))
    l1_latent_reg = self.hparams.l1_latent_lambda * self.l1_loss(z,torch.zeros_like(z))

    loss = recon_loss + l2_reg + l1_latent_reg

    # self.log('train_loss', recon_loss, prog_bar=False)
    self.log_dict({"train_loss": recon_loss, "train_l2_loss": l2_reg})
    return loss

  # def validation_step(self, x, batch_idx):
  def validation_step(self, x, batch_idx):


    x_recon, z = self.autoencoder(x)

    recon_loss = self.l2_loss(x_recon, x)
    l2_reg = self.hparams.l2_lambda * self.l2_loss(x_recon,torch.zeros_like(x_recon))
    l1_latent_reg = self.hparams.l1_latent_lambda * self.l1_loss(z,torch.zeros_like(z))
    loss = recon_loss + l2_reg + l1_latent_reg

    # self.log('val_loss', recon_loss, prog_bar=True)
    self.log_dict({"val_loss": recon_loss, "val_l2_loss": l2_reg, "l1_latent_reg":l1_latent_reg})

    return loss

  # def test_step(self, x, batch_idx):
  def test_step(self, x):

    x_recon, z = self.autoencoder(x)

    recon_loss = self.l2_loss(x_recon, x)

    l2_reg = self.hparams.l2_lambda * self.l2_loss(x_recon,torch.zeros_like(x_recon))
    l1_latent_reg = self.hparams.l1_latent_lambda * self.l1_loss(z,torch.zeros_like(z))
    pcc = pearson_corrcoef(x_recon.reshape(-1), x.reshape(-1))
    r2 = r2_score(x_recon.reshape(-1), x.reshape(-1))

    loss = recon_loss + l2_reg + l1_latent_reg
    output = {
      'mse_test': loss,
      'l2_reg': l2_reg,
      'l1_latent_reg':l1_latent_reg,
      'x_recon': x_recon,
      'z': z,
      'pcc_test': pcc,
      'r2_score': r2
    }

    self.test_step_outputs.append(output)
    # self.log('test_loss', recon_loss)
    # self.log_dict({"test_loss": recon_loss, "test_l2_loss": l2_reg})

    return {'mse_test': recon_loss, 'l2_reg':l2_reg, 'x_recon': x_recon, 'z': z, 'pcc_test':pcc, 'r2_score':r2}

  def on_test_epoch_end(self):

    mse_mean = torch.stack([x['mse_test'] for x in self.test_step_outputs]).mean()
    pcc_mean = torch.stack([x['pcc_test'] for x in self.test_step_outputs]).mean()
    r2_mean = torch.stack([x['r2_score'] for x in self.test_step_outputs]).mean()

    x_recon = torch.cat([x['x_recon'] for x in self.test_step_outputs], dim=0)
    z = torch.cat([x['z'] for x in self.test_step_outputs], dim=0)

    self.test_step_outputs.clear()
    # self.logger.experiment.add_image('Reconstructed Images', x_recon, self.current_epoch)
    # self.logger.experiment.add_embedding(z, metadata=None, global_step=self.current_epoch)

    self.log('mse', mse_mean)
    self.log('pcc', pcc_mean)
    self.log('r2', r2_mean)

    return {'test_loss': mse_mean, 'x_recon': x_recon, 'z': z}


  # def predict_step(self, x, batch_idx):
  def predict_step(self, x):

    x_recon, z = self.autoencoder(x)

    return x_recon

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
                                 # , weight_decay = 1e-8)
    lr_scheduler = {"scheduler": ReduceLROnPlateau(optimizer,...),
          "monitor": "val_loss",
          "frequency": 10,
          "verbose":True
          # If "monitor" references validation metrics, then "frequency" should be set to a
          # multiple of "trainer.check_val_every_n_epoch".
        }



    return {
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler
      }


