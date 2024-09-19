import os
import time
import tensorflow as tf
from mpoxClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
        
    @property
    def _create_tb_callbacks(self):
        # Generate a timestamp for TensorBoard logs
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")  # Fixed typo from $d to %d
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    
    @property
    def _create_ckpt_callbacks(self):
        # Set up ModelCheckpoint callback
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            save_best_only=True
        )
        
    def get_tb_ckpt_callbacks(self):
        # Return both TensorBoard and ModelCheckpoint callbacks
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]
