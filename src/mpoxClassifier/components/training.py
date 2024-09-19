import tensorflow as tf
from pathlib import Path
from mpoxClassifier.entity.config_entity import TrainingConfig
import logging

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger("mpoxClassifierLogger")
        
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
          
    def train_valid_generator(self):
        try:
            datagenerator_kwargs = dict(
                rescale=1./255,
                validation_split=0.20
            )
            
            dataflow_kwargs = dict(
                target_size=self.config.params_image_size[:-1],  # Exclude the channel dimension
                batch_size=self.config.params_batch_size,
                interpolation="bilinear"
            )
            
            valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )
            
            # Create validation generator
            self.valid_generator = valid_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="validation",
                shuffle=False,
                **dataflow_kwargs
            )
            
            # Check if we have enough samples
            if self.valid_generator.samples == 0:
                raise ValueError("No validation samples found. Check your validation split or directory path.")
            
            if self.config.params_is_augmentation:
                train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=40,
                    horizontal_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    **datagenerator_kwargs
                )
            else:
                train_datagenerator = valid_datagenerator
            
            # Create training generator
            self.train_generator = train_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="training",
                shuffle=True,
                **dataflow_kwargs
            )

            # Check if we have enough samples
            if self.train_generator.samples == 0:
                raise ValueError("No training samples found. Check your training split or directory path.")
            
            self.logger.info("Data generators created successfully.")
        except Exception as e:
            self.logger.error("Error in train_valid_generator: %s", e)
            raise
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        try:
            model.save(path)
            model.save(path.with_suffix('.keras'))  # Save using the .keras format
            logging.getLogger("mpoxClassifierLogger").info("Model saved to: %s", path)
        except Exception as e:
            logging.getLogger("mpoxClassifierLogger").error("Error saving model: %s", e)
            raise
    
    def train(self, callback_list: list):
        try:
            # Ensure we have enough samples
            if self.train_generator.samples < self.config.params_batch_size:
                raise ValueError("Not enough training samples. Consider increasing your dataset size or adjusting batch size.")
            if self.valid_generator.samples < self.config.params_batch_size:
                raise ValueError("Not enough validation samples. Consider increasing your dataset size or adjusting batch size.")
            
            self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
            self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
            
            # Add EarlyStopping and ModelCheckpoint callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', save_best_only=True, monitor='val_loss')
            ] + callback_list
            
            # Train the model
            history = self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator,
                callbacks=callbacks
            )
            
            # Save the model
            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )
            
            # Optionally, save training history
            self.logger.info("Model training completed.")
            return history
        except Exception as e:
            self.logger.error("Error during training: %s", e)
            raise
