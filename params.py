"""
Define a class for handling the hyperparameters stored in a config.ini file
===========================================================================
"""

import configparser


class Params:
    def __init__(self, path):
        self.path = path
        config = configparser.ConfigParser()
        config.read(path)
        self.config = config

        self.in_channels  = eval(config.get('model', 'in_channels'))
        self.keep_channels  = eval(config.get('model', 'keep_channels'))
        self.input_size  = eval(config.get('model', 'input_size'))
        self.output_padding  = eval(config.get('model', 'output_padding'))
        self.patch_size  = eval(config.get('model', 'patch_size'))
        self.num_patches  = eval(config.get('model', 'num_patches'))
        self.embed_dim  = eval(config.get('model', 'embed_dim'))
        self.patch_dropout  = eval(config.get('model', 'patch_dropout'))
        self.vae_latent_dim  = eval(config.get('model', 'vae_latent_dim'))
        self.sp_enc_latent_dims  = eval(config.get('model', 'sp_enc_latent_dims'))
        self.sp_dec_latent_dims  = eval(config.get('model', 'sp_dec_latent_dims'))
        self.sp_mlp_dims  = eval(config.get('model', 'sp_mlp_dims'))
        self.sp_n_heads  = eval(config.get('model', 'sp_n_heads'))
        self.sp_n_trnfr_layers  = eval(config.get('model', 'sp_n_trnfr_layers'))
        self.sp_dropouts  = eval(config.get('model', 'sp_dropouts'))

        self.epochs = eval(config.get('training', 'epochs'))
        self.batch_size = eval(config.get('training', 'batch_size'))
        self.num_workers = eval(config.get('training', 'num_workers'))
        self.learning_rate = eval(config.get('training', 'learning_rate'))
        self.learning_rate_min = eval(config.get('training', 'learning_rate_min'))
        self.save_every = eval(config.get('training', 'save_every'))

