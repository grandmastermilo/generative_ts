import torch


class RNN(torch.nn.Module):

    def __init__(self,
        input_dims:int,
        hidden_dims:int = 128,
        num_layers:int = 2,
        activation:str = 'tanh',
        use_bidirectional:bool = False,
        dropout:float = 0.1,
    ) -> None:
        """
        RNN constructor

        - creates temporal embeddings constioned on static features embedings
        - used for the generator 
        - also used for the discriminator
            - bidirectional -> linear output layer

        """
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.activation = activation
        self.use_bidirectional = use_bidirectional
        self.dropout = dropout

        self.rnn = torch.nn.RNN(
            input_size = input_dims,
            hidden_size = hidden_dims,
            num_layers = num_layers,
            nonlinearity = activation,
            dropout = dropout,
            bidirectional= use_bidirectional
        )
        return 

    def forward(self, x):
        return self.rnn(x)


class MLP:

    def __init__(self,
        input_dims:int,
        output_dims: int,
        num_layers:int = 3,
        activation:str = 'tanh',
        dropout:float = 0.1,
    ) -> None:
        """
        MLPencoder Constructor
        
        - creates static feature embeddings
        - used for decoding both temporal and static embeddings 

        """
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_layers = num_layers
        self.activation = activation
        self.dropout = dropout
        
        layers = []

        for i in range(self.num_layers):

            if i == 0:
                layers.append(torch.nn.Linear(self.input_dims, self.output_dims))
            else:
                layers.append(torch.nn.Linear(self.output_dims, self.output_dims))

            if self.activation == 'tanh':
                layers.append(torch.nn.Tanh())
            else:
                raise Exception('Not implimented')

            if dropout is not None:
                layers.append(torch.nn.Dropout(self.dropout))

        self.net = torch.nn.Sequential(*layers)

        return 


    def forward(self, x):
        return self.net(x)




class TimeGan:

    def __init__(self,
        latent_dim:int = 64, 
        nabla:float = 0.5, 
        lambd:float = 0.5
        ) -> None:
        """
        TimeGan constructor 

        - Components:
            - embedding function (encoder)
            - recovery function (decoder)
            - sequence generator 
            - seguence discriminator

        - adversial network operates in the latent space of AE

        ## embedding function - mlp (for static features), recurent net (for series)
        @param
        """

        self.latent_dim = latent_dim
        self.nabla = nabla
        self.lambd = lambd

        self.static_encoder = None
        self.temportal_encoder = None
        self.static_decoder = None
        self.temporal_decoder = None


        # for the autoencodings 
        self.static_reconstruction_loss = None
        self.temporal_reconstruction_loss = None

        # discriminator loss
        self.static_disc_loss = None
        self.temportal_disc_loss = None

        # generator loss
        self.temporal_gen_loss = None
        return 


    def train_model(self) -> None:

        return

