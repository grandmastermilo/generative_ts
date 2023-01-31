import torch


class RNNGenerator(torch.nn.Module):

    def __init__(self,
        batch_size:int,
        input_dims:int,
        hidden_dims:int,
        seq_len: int,
        num_layers:int = 2,
        activation:str = 'tanh',
        use_bidirectional:bool = False,
        dropout:float = 0.1,) -> None:
        """
        RNNGenerator constructor

        - autoregressive rnn
        - conditioned on 
        """
        super(RNNGenerator, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
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
            bidirectional= use_bidirectional,
            batch_first = True
        )

        return 

    def forward(self, x:torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the RNN generator

        - eecuted autogresively h_t = g_x(h_{0:t-1}, z_t)
        - conditioned on weiner process at each time step 

        - we must also use this generator for pass the real data for HALF PASS 
        """
        if x is None:
            #execute autoregressive generation
            
            #initial weiner conditional
            #TODO this should be a weiner process
            z_t = torch.randn((self.batch_size,self.seq_len,self.input_dims))

            #parse weiner process sequence through rnn
            gen_encodings, h_n = self.rnn(input=z_t, hx=h_n)
              
            return gen_encodings

        else:
            #execture the real data half pass

            #pass the real inputs through the generator to get generation predicitions
            preds, h_T = self.rnn(x)

            return preds



class RNNDicriminator(torch.nn.Module):

    def __init__(self,
        batch_size:int,
        input_dims:int,
        seq_len: int,
        num_layers:int = 2,
        activation:str = 'tanh',
        use_bidirectional:bool = True,
        dropout:float = 0.1,
        discriminate_context=True) -> None:
        """
        RNNGenerator constructor

        - apply a bidirection rnn over the sequences of latents (generated, real)
        """
        super(RNNDicriminator, self).__init__()

        self.batch_size = batch_size*2 # real and generated
        self.seq_len = seq_len
        self.input_dims = input_dims
        self.hidden_dims = input_dims
        self.num_layers = num_layers
        self.activation = activation
        self.use_bidirectional = use_bidirectional
        self.dropout = dropout
        self.discriminate_context = discriminate_context

        self.rnn = torch.nn.RNN(
            input_size = input_dims,
            hidden_size = input_dims,
            num_layers = num_layers,
            nonlinearity = activation,
            dropout = dropout,
            bidirectional= use_bidirectional,
            batch_first = True
        )

        self.l1 = torch.nn.Linear(self.hidden_dims*2, self.hidden_dims)
        self.l2 = torch.nn.Linear(self.hidden_dims, 1)
        
        self.tanh = torch.nn.Tanh()
        self.sig = torch.nn.Sigmoid()

        return
        
    def forward(self, latents_seq:torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the rnn dicriminator

        - use the lr and rl rnn passes final hidden state as vector to classify

        - note:
            - we could choose an alternative agregation method for h_T_LR/RL
        """

        if self.discriminate_context:
            # h_T.size -> [2*num_layers, batch_size, hidden_out_dims ]
            out, h_T = self.rnn(latents_seq) 
        
            # reshape so retreive h_out for last layers
            h_T = torch.reshape(h_T, (2, self.num_layers, self.batch_size, self.hidden_dims))

            # retreive h_t for last layers -> [2, batch_size, hidden_dims ]
            h_T = h_T[:,-1,:]

            #concatonate the h_t for each direction
            h_T = torch.cat((h_T[0], h_T[1]), dim=1)
        
            # parse the hidden states through linear layers
            x = self.tanh(self.l1(h_T))
            x = self.sig(self.l2(x))

        else:
            raise Exception("Method not yet implimented")

        return x



class RNNAutoencoder(torch.nn.Module):

    def __init__(self, 
        batch_size:int,
        input_dims:int,
        hidden_dim:int,
        seq_len: int,
        num_layers:int = 2,
        activation:str = 'tanh',
        use_bidirectional:bool = False,
        dropout:float = 0.1):
        """
        RNNAutoencoder constructor

        -
        """
        super(RNNDicriminator, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.input_dims = input_dims
        self.hidden_dims = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.use_bidirectional = use_bidirectional
        self.dropout = dropout

        self.encoder = torch.nn.RNN(
            input_size = input_dims,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            nonlinearity = activation,
            dropout = dropout,
            bidirectional= use_bidirectional,
            batch_first = True
        )
        self.decoder = torch.nn.RNN(
            input_size = hidden_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            nonlinearity = activation,
            dropout = dropout,
            bidirectional= use_bidirectional,
            batch_first = True
        )

        self.l1 = torch.nn.Linear(self.hidden_dims, self.input_dims)
        


    def encode(self,x):
        """
        Method for encoding sequence inputs to latent space
        """
        #pass the real data through the encoder
        encoding, h_T = self.encoder(x)
        return encoding

    def decode(self, x):
        """
        Mathod for decoding from latent space to input space
        """
        #pass the real latents through the decoder
        latent_decoding, h_T = self.decoder(x)

        #pass the decoded latents through the projector to return the input dims 
        decoding = self.l1(latent_decoding)

        return decoding



class TimeGan:

    def __init__(self,
        batch_size:int = 500,
        latent_dim:int = 64, 
        input_dim:int = 5,
        seq_len:int = 10,
        
        nabla:float = 0.5, 
        lambd:float = 0.5,
        static_features:list= None
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

        #intialize internal paramters
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.nabla = nabla
        self.lambd = lambd
        self.static_features = static_features # none if data set only has temporal features, list of column headers to remove from dataframe

        #REAL DATA FUNCTIONS --------------------------------------

        if static_features is not None:
            #TODO - requires dataset with static featrues
            self.static_encoder = None
            self.static_decoder = None

            self.static_genertor = None
            self.static_discriminator = None

        self.rnn_ae = RNNAutoencoder(
            batch_size=self.batch_size,
            input_dims=self.input_dim,
            hidden_dim=self.latent_dim,
            seq_len=self.seq_len,
            )

        #GENERATOR FUNCTIONs ----------------------------------

        self.temporal_generator = RNNGenerator(
            batch_size=self.batch_size,
            input_dims=self.input_dim,
            hidden_dims=self.latent_dim,
            seq_len=self.seq_len
        )

        self.temportal_discriminator = RNNDicriminator(
            batch_size=self.batch_size,
            input_dims=self.input_dim,
            hidden_dims=self.latent_dim,
            seq_len=self.seq_len
        )

        # for the autoencodings 
        #TODO check these work as expected for sequences
        self.reconstruction_loss = torch.nn.L1Loss()

        # discriminator loss
        self.gan_loss = torch.nn.BCELoss()

        # generator loss
        self.generator_loss = torch.nn.L1Loss()

        return 

    
    def forward(self, x:torch.Tensor, is_full_pass:bool) -> (torch.Tensor, torch.Tensor):
        """
        Method for the forward pass for the TimeGan

        - note discriminator receive [real latents, generate latents]
dataloading
        @param x: real sequence data
        @param is_full_pass: denotes which pass we are doing
            - full pass: AE loss, GAN loss
            - half pass: MSE loss 
        """

        if is_full_pass:
            # encode latents and generate latents
            real_latents = self.rnn_ae.encode(x)
            gen_latents = self.temporal_generator.forward()

            # concatonate tensors to pass to discriminator
            disc_inputs = torch.cat([real_latents, gen_latents], dim=0)
            
            # get discriminator classification
            disc_classification = self.temportal_discriminator(disc_inputs)

            # get reconstruction from real latents
            real_reconstruction = self.rnn_ae.decode(real_latents)

            return disc_classification, real_reconstruction

        else:
            # get real and generated latents from real data
            real_latents = self.rnn_ae.encode(x)
            gen_latents = self.temporal_generator.forward(x)
            
            
            return real_latents, gen_latents





if __name__ == "__main__":

    # print('TESTING RNN GENERATOR -- AUTOREGRESSIVE')
    
    # input_dims = 5
    
    # rnn_gen = RNNGenerator(batch_size=3, input_dims=5, seq_len=3)

    # rnn_gen.forward()

    # print('TESTING RNN GENERATOR -- PREDICTIVE')
    
    # input_dims = 5
    # batch_size = 2
    # seq_len = 3

    # inputs = torch.randn((batch_size, seq_len, input_dims))
    
    # rnn_gen = RNNGenerator(batch_size=batch_size, input_dims=input_dims, seq_len=seq_len)

    # rnn_gen.forward(inputs)

    # print('TESTING RNN DISCRIMINATOR ')
    
    # input_dims = 5
    # batch_size = 2
    # seq_len = 3

    # inputs = torch.randn((batch_size*2, seq_len, input_dims))
    
    # rnn_disc = RNNDicriminator(batch_size=batch_size, input_dims=input_dims, seq_len=seq_len)

    # rnn_disc.forward(inputs)




    # print('TESTING TIMEGAN MODEL :')

    # model = TimeGan(
    #     latent_dim=64,
    #     projection_dim=256
    # )