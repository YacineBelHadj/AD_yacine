
def main():
    input_data,time=get_processed_PSD()
    print(input_data.shape)
    train_data = input_data[:int(0.8*len(input_data))]


if __name__ == '__main__':
    main()
