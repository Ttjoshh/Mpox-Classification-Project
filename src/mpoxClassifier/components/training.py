from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
from pathlib import Path
from mpoxClassifier.entity.config_entity import TrainingConfig
import logging



class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger("mpoxClassifierLogger")
        
    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
          
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
            
            self.valid_generator = valid_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="validation",
                shuffle=False,
                **dataflow_kwargs
            )
            
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
                    brightness_range=[0.8, 1.2],  # Randomly change brightness
                    fill_mode='nearest',
                    **datagenerator_kwargs
                )
            else:
                train_datagenerator = valid_datagenerator
            
            self.train_generator = train_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="training",
                shuffle=True,
                **dataflow_kwargs
            )
            
            if self.train_generator.samples == 0:
                raise ValueError("No training samples found. Check your training split or directory path.")
            
            self.logger.info(f"Training samples: {self.train_generator.samples}, Validation samples: {self.valid_generator.samples}")

            self.logger.info("Data generators created successfully.")
        except Exception as e:
            self.logger.error("Error in train_valid_generator: %s", e)
            raise
    
    def calculate_class_weights(self):
        try:
            # Get the total number of samples per class from the training generator
            labels = self.train_generator.classes 
            if len(labels) == 0:
                raise ValueError("No labels found in the training generator.")# List of class indices for each image
            
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(labels),
                y=labels
            )
            class_weight_dict = dict(enumerate(class_weights))  # Map class index to weight
            
            self.logger.info("Class weights calculated successfully: %s", class_weight_dict)
            return class_weight_dict
        except Exception as e:
            self.logger.error("Error in calculating class weights: %s", e)
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
            if self.train_generator.samples < self.config.params_batch_size:
                raise ValueError("Not enough training samples. Consider increasing your dataset size or adjusting batch size.")
            if self.valid_generator.samples < self.config.params_batch_size:
                raise ValueError("Not enough validation samples. Consider increasing your dataset size or adjusting batch size.")
            
            self.steps_per_epoch = max(1, self.train_generator.samples // self.train_generator.batch_size)
            self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
            
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate, clipnorm=1.0)
            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


             # Initialize callbacks
            callbacks = []  # Initialize the callbacks list here
            
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            )
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(filepath='best_model.keras', save_best_only=True, monitor='val_loss')
            )
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
            )
            callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)
            )

            # Add any user-defined callbacks
            callbacks.extend(callback_list)
            
           

            # Calculate class weights
            class_weight_dict = self.calculate_class_weights()
            
            callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)
            )

            # Train the model with class weights
            history = self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator,
                callbacks=callbacks,
                class_weight=class_weight_dict  # Pass the calculated class weights
            )
            
            # Save the model
            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )
            
            self.logger.info("Model training completed.")
            return history
        except Exception as e:
            self.logger.error("Error during training: %s", e)
            raise
