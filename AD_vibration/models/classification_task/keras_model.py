from keras.layers import Input, Dense 
from keras.models import Model

def Dense_classification(inputDim:tuple,num_class:int):
    """ Dense based classification model
    Parameters
    ----------
    inputDim : tuple
        Input dimension
        num_class : int 
        Number of classes to classify
"""
    inputLayer = Input(shape=(inputDim))
    h = Dense(100,activation='relu')(inputLayer)
    h = Dense(100,activation='relu')(h)
    emb = Dense(50,activation='relu')(h)
    out = Dense(num_class,activation='softmax')(emb)

    model = Model(inputs=inputLayer, outputs=out)
    encoder= Model(inputs=inputLayer, outputs=emb)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model,encoder

