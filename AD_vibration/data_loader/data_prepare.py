
import numpy as np
import sys
from sklearn.preprocessing import OneHotEncoder
from pydantic import BaseModel, validator

EPS= sys.float_info.epsilon


class DataPrepare:
    labels_options = {12 : np.array([['2X','2Y','2Z','3X','3Y','3Z','4X','4Y','4Z','5X','5Y','5Z']]),
                        3 : np.array([['X','Y','Z','X','Y','Z','X','Y','Z','X','Y','Z']])}
    num_sensor = 12

    def __init__(self,num_classes:int,training_length:int,apply_log:bool):
        self.num_classes = num_classes
        self.training_length = training_length
        self.apply_log = apply_log

    @validator('num_classes')
    def num_classes_validations(cls,v):
        if v not in self.labels_options.keys():
            raise ValueError('Number of classes not supported')
        return v

    def encode_label(self,label:np.ndarray,training=False):
        if training:
            self.OneHotEncoder = OneHotEncoder(handle_unknown='ignore')
        self.OneHotEncoder.fit(label.reshape(-1,1))
        return self.OneHotEncoder.transform(label.reshape(-1,1)).toarray()
        
    def prepare_data(self,input_data:np.ndarray,return_data=False):
        if self.apply_log:
            input_data = np.log10(input_data+EPS)

        labels = self.labels_options[self.num_classes]    
        labels=np.repeat(labels,len(input_data),axis=0)
        training_data = input_data[:self.num_sensor*self.training_length,:,:]

        self.length_input_signal = input_data.shape[-1]
        self.min_training = np.min(training_data,axis=(0,2))
        self.max_training = np.max(training_data,axis=(0,2))
        
        input_data = (input_data -self.min_training[None,:,None])/(self.max_training[None,:,None]-self.min_training[None,:,None])
        X = input_data.reshape((-1,self.length_input_signal))
        Y = labels.reshape((-1))

        X_train = X[:self.num_classes*self.training_length]
        y_train = Y[:self.num_classes*self.training_length]

        y_train=self.encode_label(y_train,training=True)
        if return_data:
            return X_train,y_train,X,Y, input_data
        return X_train,y_train,X,Y

if __name__ == '__main__':
    data_prepare= DataPrepare(12,1000,True)
    input_data = np.random.rand(3000,12,370)
    X_train,y_train,X,Y= data_prepare.prepare_data(input_data)
    print(X_train.shape)
    print(y_train.shape)
    print(data_prepare.min_training.shape)