from mpoxClassifier.config.configuration import ConfigurationManager
from mpoxClassifier.components.prepare_callbacks import PrepareCallback
from mpoxClassifier.components.training import Training
from mpoxClassifier import logger

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            # Load configuration for callbacks
            config = ConfigurationManager()
            prepare_callbacks_config = config.get_prepare_callback_config()
            prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
            callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
            
            # Load training configuration
            training_config = config.get_training_config()
            
            # Initialize Training class with the configuration
            training = Training(config=training_config)
            
            # Load and prepare the EfficientNetB5 base model
            training.get_base_model()  # This will load the base model and add the new output layer
            
            # Create data generators for training and validation
            training.train_valid_generator()
            
            # Train the model with the prepared callbacks
            training.train(callback_list=callback_list)
            
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        
        except Exception as e:
            logger.exception(e)
            raise e
        
if __name__ == '__main__':
    try:
        logger.info(f"************************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.exception(e)
        raise e
