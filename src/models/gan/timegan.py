import torch


class RNN:

    def __init__(self) -> None:
        """
        RNN constructor

        - creates temporal embeddings constioned on static features embedings
        - used for the generator 
        - also used for the discriminator
            - bidirectional -> linear output layer

        """
        return 


class MLP:

    def __init__(self) -> None:
        """
        MLPencoder Constructor
        
        - creates static feature embeddings
        - used for decoding both temporal and static embeddings 

        """
        return 



class TimeGan:

    def __init__(self) -> None:
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

