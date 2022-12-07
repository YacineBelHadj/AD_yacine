from keras.layers import Input, Dense, LSTM
from keras.models import Model
from  AD_vibration.models.classification_task.keras_layers import CosFace

def LSTM_classification(inputDim:tuple,num_class:int,cosface_loss=False):
    """ LSTM based classification model
    Parameters
    ----------
    inputDim : tuple
        Input dimension
        num_class : int 
        Number of classes to classify
        cosface_loss : bool (default=False)"""
    inputLayer = Input(shape=(inputDim))
    labels = Input(shape=(num_class,))

    h = LSTM(100)(inputLayer)
    h = Dense(100,activation='relu')(h)
    emb = Dense(50,activation='relu')(h)

    if cosface_loss:
        out = CosFace(num_class)([emb,labels])
        model = Model(inputs=[inputLayer,labels], outputs=out)
    else : 
        out = Dense(num_class,activation='softmax')(emb)
        model = Model(inputs=inputLayer, outputs=out)

    model_pred = Model(inputs=inputLayer, outputs=out)
    encoder= Model(inputs=inputLayer, outputs=emb)
    
    return model,model_pred,encoder
