from keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, ReLU,DepthwiseConv1D, add
from AD_vibration.models.classification_task.keras_layers import CosFace
from keras.models import Model

def _expansion_block(x,t,filters,block_id):
    prefix = f'block_{block_id}_'
    total_filters = t * filters
    x=Conv1D(total_filters,1,strides=1,padding='same',name=prefix+'expand')(x)
    x=BatchNormalization(name=prefix+'expand_BN')(x)
    x=ReLU(name=prefix+'expand_relu')(x)
    return x

def _depthwise_block(x,kernel_size,strides,block_id):
    prefix = f'block_{block_id}_'
    x=DepthwiseConv1D(kernel_size=kernel_size,strides=strides,padding='same',name=prefix+'depthwise')(x)
    x=BatchNormalization(name=prefix+'depthwise_BN')(x)
    x=ReLU(name=prefix+'depthwise_relu')(x)
    return x

def _projection_block(x,out_channels,block_id):
    prefix = f'block_{block_id}_'
    x = Conv1D(filters=out_channels,kernel_size=1,padding='same',use_bias=False,name=prefix+'compress')
    x= BatchNormalization(name=prefix+'compress_BN')(x)
    return x

def Bottleneck(x,t,filters,kernel_size,out_channels,stride,block_id):
    y = _expansion_block(x,t,filters,block_id)
    y = _depthwise_block(y,kernel_size,stride,block_id)
    y = _projection_block(y,out_channels,block_id) 
    if y.shape[-1]==x.shape[-1]:
        y = add([x,y])
    return y


def MobileNetV2(input_shape = (198,1), n_classes=12, CosFace:bool = True):    
    inputLayer = Input(input_shape)
    label = Input(shape=(n_classes,))

    x = Conv1D(32,3,strides=2,padding='same',activation='relu')(inputLayer)
    x = Conv1D(32,3,strides=2,padding='same',activation='relu')(x)  
    x = Bottleneck(x, t = 2, filters = x.shape[-1],kernel_size=3, out_channels = 128, stride = 2,block_id = 2)    
    x = Bottleneck(x, t = 4, filters = x.shape[-1],kernel_size=3, out_channels = 128, stride = 1,block_id = 3)    
    x = Bottleneck(x, t = 4, filters = x.shape[-1],kernel_size=3, out_channels = 128, stride = 2,block_id = 4)    

    x = Conv1D(filters = 512,kernel_size = 1,padding='same',use_bias=False, name = 'pre_final_conv')(x)
    x = BatchNormalization(name='last_bn')(x)
    x = ReLU(name='last_relu')(x)    

    x = Conv1D(filters = 512,kernel_size = 1,padding='same',use_bias=False, activation=None, name = 'conv_lin')(x)
    x = Conv1D(filters = 128,kernel_size = 1,padding='same',use_bias=False,activation = None, name = 'last_conv_lin_')(x)
    x = Dropout(0.2)(x)
    emb = GlobalAveragePooling1D()(x)
    if CosFace:
        out = CosFace(n_classes)([emb,label])
        model_train = Model(inputs=[inputLayer,label], outputs=out)
    else :
        out = Dense(n_classes)(emb)
        model_train = Model(inputs=[inputLayer,label], outputs=out)

    encoder = Model(inputs=inputLayer, outputs=emb) 
    return model, encoder

if __name__=="__main__":
    import tensorflow as tf
    mobile=tf.keras.applications.mobilenet_v2.MobileNetV2()
    mobile.summary()