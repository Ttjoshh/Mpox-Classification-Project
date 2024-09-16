from mpoxClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf 
from pathlib import Path 


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def get_base_model(self):
        # Load the base model from the specified path
        base_model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # Remove the last layer (the Dense layer with 2 units)
        base_model.layers.pop()

        # Get the output of the previous layer (after flatten)
        x = base_model.layers[-1].output

        # Add a new Dense layer with 6 units for the 6 classes
        output_layer = tf.keras.layers.Dense(6, activation='softmax')(x)
        
        # Create a new model that includes the base model and the new output layer
        self.model = tf.keras.Model(inputs=base_model.input, outputs=output_layer)

        # Compile the model with appropriate loss and optimizer
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss='categorical_crossentropy',  # Assuming one-hot encoded labels
            metrics=['accuracy']
        )


        
    def train_valid_generator(self):
        
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )
        
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
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
            
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        
    def train(self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )
        
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )