import cv2
import os
import numpy as np
from model import Model

def get_dataset(input_folder, image_size=50):
    labels = []
    X = []
    for i, name in enumerate(os.listdir(input_folder)):
        sub_folder = os.path.join(input_folder, name)
        labels.extend([i]*len(os.listdir(sub_folder)))
        for file_name in os.listdir(sub_folder):
            file_path = os.path.join(sub_folder, file_name)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (image_size, image_size))
            x = image.reshape(-1)
            X.append(x)
    X = np.array(X) / 255.0
    labels = np.array(labels).reshape(-1)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    labels_shuffled = labels[indices]
    return X_shuffled,labels_shuffled

def get_data_generator(root_data=r"C:\Users\D E L L\Downloads\nam_code\dataset\dataset_extracted",
                       image_size=50,
                       batch_size=16):
    X, labels = get_dataset(root_data, image_size)
    
    def generator():
        while True:
            # Shuffle the data at the beginning of each epoch
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            labels_shuffled = labels[indices]
            
            for start in range(0, X.shape[0], batch_size):
                end = min(start + batch_size, X.shape[0])
                yield X_shuffled[start:end].T, np.expand_dims(labels_shuffled[start:end],0)
    
    return generator


epochs = 50
if __name__ == "__main__":
    model = Model([2500,1200,1])
    data_gen = get_data_generator(batch_size=64)
    model.fit(data_gen,epochs=epochs)
    






















































































































import nn
import numpy as np
from optimizer import optimizer
def test_run():
    """
    Sample test run.

    :return: None
    """
    
    # test run for binary classification problem:
    np.random.seed(3)
    print('Running a binary classification test')

    #Generate sample binary classification data
    x, y = get_dataset(r"dataset\dataset_extracted")
    x_train = x[:60,:]
    y_train = y[:60]
    x_test = x[60:,:].T
    y_test = y[60:]
    
    X = x_train.T
    Y = np.expand_dims(y_train,axis=0)
    # import pdb;pdb.set_trace()
    net = nn.nn([2500,1200,1],['relu','sigmoid'])
    net.cost_function = 'CrossEntropyLoss'
    optim = optimizer.SGDOptimizer
    optim(X,Y,net,128,alpha=0.07,epoch=epochs,lamb=0.05,print_at=1,momentum=0.9)

   
test_run()

