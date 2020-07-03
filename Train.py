"""
"""
from tensorflow.keras.layers import Input, Lambda, concatenate, \
    Conv3D, Dropout, MaxPooling3D, Conv3DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_core.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from SharedMethods import get_organized_data


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


def train(SAVE_PATH, DIMENSIONS, ORGAN, val_split=0.1, batch_size=15, epochs=50):
    # get training data
    path_x_train_resampled = "{}Xtrain/resampled/".format(SAVE_PATH)
    path_y_train_resampled = "{}ytrain/resampled/".format(SAVE_PATH)
    x_train = get_organized_data(path_x_train_resampled, DIMENSIONS)
    y_train = get_organized_data(path_y_train_resampled, DIMENSIONS, True)

    # generate the U-Net model (Width, Height, Depth, Channels)
    architecture = generate_U_Net(DIMENSIONS[0], DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[3])

    # train U-Net on training data and save it
    # set parameters for training and train the model
    tensorboard = TensorBoard(histogram_freq=1)
    earlystopper = EarlyStopping(patience=20, verbose=1)
    checkpointer = ModelCheckpoint('{}{}U-Net.h5'.format(SAVE_PATH, ORGAN), verbose=1, save_best_only=True)
    history = architecture.fit(x_train, y_train,
                        validation_split=val_split,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[earlystopper, checkpointer, tensorboard])

    # generate image with model architecture and show training history
    plot_model(architecture, to_file='{}U-Net.png'.format(SAVE_PATH), show_shapes=True)
    plot_history(history)

    #INFO: save U-Net non needed for early stopping already saves the best model
    #model.save('{}U-Net.h5'.format(save_path))
