import tensorflow as tf

### Define the main layer

class Conv_1D_Block(tf.keras.layers.Layer):
    def __init__(self,filter,kernel):
      super().__init__()
      self.conv = tf.keras.layers.Conv1D(filter,kernel,padding='same',kernel_initializer='he_normal')
      self.bn = tf.keras.layers.BatchNormalization()
      self.activation = tf.keras.layers.Activation('relu')
    def call(self,inputs,training=False):
      x = self.conv(inputs)
      x = self.bn(x)
      x = self.activation(x)
      return x

### Define the modelS

class Cnn_classifier(tf.keras.Model):
  
  def __init__(self,input_dim=(3701,1),conv_layer_filter = [64,128,256,512],kernel=5,dense_layer = [1024,512], num_classes=12):
    super().__init__()
    self.input_dim = input_dim
    self.conv_layer = conv_layer_filter
    self.dense_layer = dense_layer
    self.num_classes = num_classes

    self.conv_layer = [Conv_1D_Block(filter,kernel) for filter in conv_layer_filter]
    self.dense_layer = [tf.keras.layers.Dense(units,activation='relu') for units in dense_layer]

  def call_conv(self, inputs, training=False):
    x = inputs
    for conv in self.conv_layer:
      x = conv(x)
      x = tf.keras.layers.MaxPool1D()(x)
    return x

  def call_dense(self, inputs, training=False):
    x = inputs
    for dense in self.dense_layer:
      x = dense(x)
    return x

  def call(self, inputs, training=False):
    x = self.call_conv(inputs)
    x = self.call_dense(x)
    return x

  def build_model(self):
    x = tf.keras.layers.Input(shape=self.input_dim)
    self.encoder = tf.keras.Model(inputs=[x], outputs=tf.keras.layers.Flatten()(self.call_conv(x)))
    self.model = tf.keras.Model(inputs=[x], outputs=self.call_dense(self.encoder(x)))
    self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

