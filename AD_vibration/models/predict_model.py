import tensorflow as tf

class Classification_model():
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.model.summary()
    def forward