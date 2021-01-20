def load_cifar10(DIR):
    """
    Input  - path to cifar10 directory
    Output - X_train, y_train, X_test, y_test
    """
    import os
    import pickle
    import numpy as np

    img_data, img_label = [], []
    for batch in range(1,6):
        with open(os.path.join(DIR, f'data_batch_{batch}'), mode='rb') as file:
            data = pickle.load(file, encoding='latin1')

            #Each batch has array with 10000 rows, each row of the array stores a 32x32 colour image.
            #The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
            #The image is stored in row-major order,
            #so that the first 32 entries of the array are the red channel values of the first row of the image.
            imgs = data['data']
            imgs = np.reshape(a=imgs, newshape=(10000, 3, 32, 32), order='C').transpose(0, 2, 3, 1)

            img_data.append(imgs)
            img_label.append(data['labels'])

    X_train = np.concatenate(img_data, axis=0)
    y_train = np.concatenate(img_label, axis=0)

    with open(os.path.join(DIR, 'test_batch'), mode='rb') as file:
        data = pickle.load(file, encoding='latin1')

        imgs = data['data']
        imgs = np.reshape(a=imgs, newshape=(10000, 3, 32, 32), order='C').transpose(0, 2, 3, 1)

        X_test = imgs
        y_test = np.array(data['labels'])

    return X_train, y_train, X_test, y_test
