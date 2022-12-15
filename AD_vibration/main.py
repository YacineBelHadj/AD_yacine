#%%
from AD_vibration.data_loader.data_prepare import DataPrepare
from AD_vibration.data_loader.data_loader import get_processed_PSD
from AD_vibration.visualization.visualize import plot_example_psd
from AD_vibration.models.track.tr_tensorboard import get_callbacks
from AD_vibration.models.classification_task.keras_model import DenseSignalClassifier

import matplotlib.pyplot as plt

#%% 
input_data,time=get_processed_PSD()
input_data = input_data[:,2:,:]
dp= DataPrepare(num_classes=12,training_length=1500,apply_log=True)
X_train,y_train,X,Y= dp.prepare_data(input_data)
#%%

model = DenseSignalClassifier(inputDim=X_train[0].shape, num_class=12,dense_layers=[2048,1024,512,256,128,64,32]).build_model()
model.fit(X_train,y_train,epochs=100,batch_size=32,callbacks=get_callbacks(),validation_split=0.2)


# %%

