from AD_vibration.data_loader.data_prepare import DataPrepare
from AD_vibration.data_loader.data_loader import get_processed_PSD
from AD_vibration.visualization.visualize import plot_example_psd
import matplotlib.pyplot as plt

def main():
    input_data,time=get_processed_PSD()
    input_data = input_data[:,2:,:]
    dp= DataPrepare(num_classes=12,training_length=1500,apply_log=True)
    X_train,y_train,X,Y= dp.prepare_data(input_data)
    print(X_train[0].shape)
    for i in range(20):
        plot_example_psd(X_train[i],fs=125)
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    main()
