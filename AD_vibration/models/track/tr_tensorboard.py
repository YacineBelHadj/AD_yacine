import tensorflow as tf
from datetime import datetime

def get_callbacks():
    logdir_tb = "./logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir_model= "./logs/model/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Create a TensorBoard callback that will log the given log directory
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir_tb)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=logdir_model,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    return [tensorboard_callback,model_checkpoint_callback]