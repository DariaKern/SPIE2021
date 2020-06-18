"""
METHODS:
    generate the U-Net architecture
    train Model
    plot -training history


generate_U-Net():
    Source: /home/daria/Desktop/Data/oOoMietzner/Share/Studierendentagung/U-Net/U-Net_Generator.py
    Changes:
        -2D -> 3D
        -add IMG_DEPTH

"""
from tensorflow.keras.layers import Input, Lambda, concatenate, \
    Conv3D, Dropout, MaxPooling3D, Conv3DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf


# generate the U-Net architecture and return it
from tensorflow_core.python.keras.callbacks import EarlyStopping, ModelCheckpoint


def generate_U_Net(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, IMG_CHANNELS):
    # Build U-Net model
    inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    #u9 = concatenate([u9, c1], axis=3)
    u9 = concatenate([u9, c1], axis=4)

    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])

    # optimizer=keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    model.summary()

    return model


def train_U_Net(model, X_train, y_train, save_path, val_split=0.1, batch_size=15, epochs=50):
    # set parameters for training and train the model
    earlystopper = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint('{}U-Net.h5'.format(save_path), verbose=1, save_best_only=True)
    history = model.fit(X_train, y_train,
                        validation_split=val_split,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[earlystopper, checkpointer])

    # generates image with model architecture
    plot_model(model, to_file='{}U-Net.png'.format(save_path), show_shapes=True)

    #INFO: save U-Net non needed for early stopping already saves the best model
    #model.save('{}U-Net.h5'.format(save_path))

    return model, history


def plot_history(history):
    # Plot history as image: Binary crossentropy & Accuracy
    plt.plot(history.history['loss'], label='binary crossentropy (training data)')
    plt.plot(history.history['val_loss'], label='binary crossentropy (validation data)')
    plt.plot(history.history['accuracy'], label='Accuracy (training data)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
    plt.title('Model performance')
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

