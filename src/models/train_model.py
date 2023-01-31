from models.gan.timegan import TimeGan

from features.loaders.stock_dataset import StocksDataset


class ModelTrainer:

    def __init__(self, 
        model_type:str, 
        model_kwargs:dict,
        
        dataset_type:str,
        dataset_kwargs:dict,

        training_iterations:int
        ) -> None:
        """
        ModelTrainer constuctor

        class for oractrsting training of all generative models
        
        """

        #initialize internal variables
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.dataset_type = dataset_type
        self.dataset_kwargs = dataset_kwargs

        self.training_iterations = training_iterations
    

        #intializers places holder, to be initialized
        self.model = None
        self.dataset = None

        if self.model_type == 'timegan':
            self.model = TimeGan(**self.model_kwargs)
        else:
            raise Exception("model type not available")


        if self.dataset_type == 'stocks':
            self.dataset = StocksDataset(**self.dataset_kwargs)
        else:
            raise Exception("dataset type not available")

        return


    def training_step(self) -> None:
        """
        Method for implimenting the training process for the algorthim
        """

        if self.model_type == 'timegan':

            #step 1 --- full pass


            #step 2 --- half pass

            pass
        else:
            raise Exception("model type not available")

        return 