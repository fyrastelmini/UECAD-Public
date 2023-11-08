from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig,AdversarialTrainerConfig
import pythae.models as pm
import torch.nn as nn
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput

def get_model(name,dim,train_batch,lstm=False,len_seq=24*60,latent_dim=256):
    class Encoder_LSTM_VAE(BaseEncoder):
        def __init__(self, args):
            BaseEncoder.__init__(self)

            self.input_dim = args.input_dim
            self.latent_dim = args.latent_dim
            _,self.channels,_ = self.input_dim
            # Define the LSTM layers
            self.lstm_layers = nn.Sequential(
                nn.LSTM(input_size=24*self.channels, hidden_size=128, num_layers=2, batch_first=True)
            )
            self.relu = nn.ReLU()
            # Linear layers for mean and log variance
            self.embedding = nn.Linear(128, self.latent_dim)
            self.log_var_layer = nn.Linear(128, self.latent_dim)

        def forward(self, x):
            # Reshape the input
            x = x.view(x.size(0), 1, 24*self.channels)

            # Pass through LSTM layers
            lstm_output, _ = self.lstm_layers(x)

            # Take the last output
            lstm_output = lstm_output[:, -1, :]
            # Apply ReLU activation
            lstm_output = self.relu(lstm_output)
            # Calculate mean and log variance
            output = ModelOutput(
            embedding = self.embedding(lstm_output),
            log_covariance = self.log_var_layer(lstm_output))

            return output

    class Decoder_LSTM_AE(BaseDecoder):
        def __init__(self, args):
            BaseDecoder.__init__(self)

            self.input_dim = args.input_dim
            self.latent_dim = args.latent_dim
            _,self.channels,_ = self.input_dim

            # Linear layer to map from latent to LSTM input
            self.fc = nn.Linear(self.latent_dim, 24*self.channels)

            # Define the LSTM layers
            self.lstm_layers = nn.Sequential(
                nn.LSTM(input_size=24*self.channels, hidden_size=128, num_layers=2, batch_first=True)
            )
            self.relu = nn.ReLU()
            # Linear layer to map from LSTM output to reconstruction
            self.reconstruction_layer = nn.Linear(128, 24*self.channels)

        def forward(self, z):
            # Pass through linear layer
            lstm_input = self.fc(z)

            # Expand dimensions to match LSTM input shape
            lstm_input = lstm_input.view(z.size(0), 1, 24*self.channels)

            # Pass through LSTM layers
            lstm_output, _ = self.lstm_layers(lstm_input)

            # Reshape for the final reconstruction
            lstm_output = lstm_output.view(z.size(0), -1)
            lstm_output = self.relu(lstm_output)
            # Pass through the reconstruction layer
            output= ModelOutput(
            reconstruction = self.reconstruction_layer(lstm_output))

            return output

    # Set up the training configuration
    my_training_config = BaseTrainerConfig(output_dir=None,num_epochs=50,learning_rate=1e-3,no_cuda=False,per_device_train_batch_size=train_batch,per_device_eval_batch_size=1,train_dataloader_num_workers=1,
    eval_dataloader_num_workers=1,steps_saving=300,optimizer_cls="AdamW",optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 5, "factor": 0.5})
    
    if name=="AE":
        model_config = pm.AEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.AE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    
    elif name=="VAE":
        # Set up the model configuration 
        model_config = pm.VAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim)
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None
        # Build the model
        my_vae_model = pm.VAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder)
    elif name=="BETA":   
        model_config = pm.BetaVAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        beta=1.5)
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None
        my_vae_model = pm.BetaVAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name ==  "VAE_LinNF":
        model_config = pm.VAE_LinNF_Config(
            input_dim=(1, dim, len_seq),
            latent_dim=latent_dim,
            flows=['Planar', 'Radial', 'Planar']
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None
        my_vae_model = pm.VAE_LinNF(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="VAE_IAF":
        model_config = pm.VAE_IAF_Config(
            input_dim=(1, dim, len_seq),
            n_made_blocks=2,
            n_hidden_in_made=3,
            hidden_size=latent_dim)
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None
        my_vae_model = pm.VAE_IAF(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder)
    elif name== "DBVAE":
        model_config = pm.DisentangledBetaVAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        beta=5.,
        C=30.0,
        warmup_epoch=25)
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None
        my_vae_model = pm.DisentangledBetaVAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="FactorVAE":
        model_config = pm.FactorVAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        gamma=10.
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None
        my_vae_model = pm.FactorVAE(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="BETA-TC":
        model_config = pm.BetaTCVAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        beta=2.,
        alpha=1,
        gamma=1)
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None
        my_vae_model = pm.BetaTCVAE(model_config=model_config,
        encoder=encoder,
        decoder=decoder)
    elif name=="IWVAE":
        model_config = pm.IWAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        number_samples=3
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.IWAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="MIWAE":
        model_config = pm.MIWAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        number_gradient_estimates=4,
        number_samples=4)
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.MIWAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="PIWAE":
        model_config = pm.PIWAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        number_gradient_estimates=4,
        number_samples=4,
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.PIWAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="CIWAE":
        model_config = pm.CIWAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        beta=0.05,
        number_samples=4
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.CIWAE(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="MISSMVAE":
        model_config = pm.MSSSIM_VAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        beta=1e-2,
        window_size=3)
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.MSSSIM_VAE(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )   
    elif name=="WAE":
        model_config = pm.WAE_MMD_Config(
            input_dim=(1, dim, len_seq),
            latent_dim=latent_dim,
            kernel_choice='imq',
            reg_weight=100,
            kernel_bandwidth=2)
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.WAE_MMD(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder)
    elif name =="INFOVAE":
        model_config = pm.INFOVAE_MMD_Config(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        kernel_choice='imq',
        alpha=-2,
        lbd=10,
        kernel_bandwidth=1
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.INFOVAE_MMD(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="VAMP":
        model_config = pm.VAMPConfig(
            input_dim=(1, dim, len_seq),
            latent_dim=latent_dim,
            beta=1e-3,
            window_size=2)
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.VAMP(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder)
    elif name=="SVAE":
        model_config = pm.SVAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.SVAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="PVAE":
        model_config = pm.PoincareVAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        reconstruction_loss="bce",
        prior_distribution="riemannian_normal",
        posterior_distribution="wrapped_normal",
        curvature=0.7
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.PoincareVAE(
        model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="VQVAE":
        model_config = pm.VQVAEConfig(
        latent_dim=latent_dim,
        input_dim=(1, dim, len_seq),
        commitment_loss_factor=0.25,
        quantization_loss_factor=1.0,
        num_embeddings=128,
        use_ema=True,
        decay=0.99
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.VQVAE(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="HVAE":
        model_config = pm.HVAEConfig(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        n_lf=1,
        eps_lf=0.001,
        beta_zero=0.3,
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.HVAE(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="RAE_GP":
        model_config = pm.RAE_GP_Config(
        input_dim=(1, dim, len_seq),
        latent_dim=latent_dim,
        embedding_weight=1e-2,
        reg_weight=1e-4
        )
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.RAE_GP(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder
        )
    elif name=="RHVAE":
        model_config = pm.RHVAEConfig(
            input_dim=(1, dim, len_seq),
            latent_dim=latent_dim,
            n_lf=1,
            eps_lf=0.001,
            beta_zero=0.3,
            temperature=1.5,
            regularization=0.001)
        if lstm:
            encoder = Encoder_LSTM_VAE(model_config)
            decoder= Decoder_LSTM_AE(model_config)
        else:
            encoder = None
            decoder = None

        my_vae_model = pm.RHVAE(
            model_config=model_config,
        encoder=encoder,
        decoder=decoder)
     # Build the Pipeline
    pipeline = TrainingPipeline(training_config=my_training_config,
    model=my_vae_model)
    return pipeline
