import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import BatchNormalization, LeakyReLU

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from cordex_discrete import cordex_discrete


def create_train_val_set(runs, n_x, scalars, optimality, J_cb,
                         cordex=cordex_discrete, num_designs=1000, test_size=0.2, random_state=42, noise=None):
    """
    Generate design matrices using the cordex_discrete function and split the resulting data into
    training and validation sets.

    Args:
        runs (int): The number of runs for the cordex_discrete function.
        n_x (list): The list of variables for the cordex_discrete function.
        scalars (list): The list of scalars for the cordex_discrete function.
        optimality (str): The optimality criteria for the cordex_discrete function.
        J_cb (matrix): The matrix for the cordex_discrete function.
        cordex (function, optional): The cordex function to use. Defaults to cordex_discrete.
        num_designs (int, optional): The number of design matrices to generate. Defaults to 1000.
        test_size (float, optional): The proportion of the dataset to be used as the validation set. Defaults to 0.2.
        random_state (int, optional): The random seed for reproducible dataset splitting. Defaults to 42.
        noise (float, optional): The standard deviation of the Gaussian noise to add to the design matrices.
    Returns:
        tuple: A tuple containing the training and validation sets as NumPy arrays:
            - train_data (np.array): The training set.
            - val_data (np.array): The validation set.
    """
    # Generate design matrices using cordex_discrete
    design_matrices = []
    criteria_matrix = []
    for _ in tqdm(range(num_designs)):
        opt_design, opt_cr = cordex(runs=runs, f_list=n_x, scalars=scalars, levels=[-1, 1], epochs=1,
                                    optimality=optimality, J_cb=J_cb, disable_bar=True)
        design_matrices.extend(opt_design)
        criteria_matrix.append(opt_cr)

    design_matrices = np.array(design_matrices)
    # Normalize the design matrices
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_designs = scaler.fit_transform(design_matrices.reshape(num_designs, -1))
    if noise is not None:
        # Add noise to the design matrices
        noisy_designs = normalized_designs + noise * np.random.normal(size=normalized_designs.shape)
        noisy_designs = np.clip(noisy_designs, -1, 1)

        # Split the data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(noisy_designs, normalized_designs, test_size=test_size,
                                                          random_state=random_state)
        return x_train, x_val, y_train, y_val
    else:
        # Split the data into training and validation sets
        train_data, val_data = train_test_split(normalized_designs, test_size=test_size, random_state=random_state)
        return train_data, val_data


def create_autoencoder(input_dim, latent_dim, dropout_rate=0.1,
                       latent_space_activation='tanh', output_activation='tanh'):
    """
    Create an autoencoder with the given parameters. The autoencoder consists of an encoder and a decoder.
    The encoder compresses the input data into a lower-dimensional latent space, and the decoder reconstructs
    the original data from the latent space representation. The number of encoder and decoder layers and their
    neuron counts are determined by the input_dim and latent_dim parameters.

    The architecture of the autoencoder is dynamically constructed based on the input parameters. Specifically,
    the number of layers in the encoder and decoder is determined by the relationship between the input_dim
    and latent_dim. The encoder layers consist of Dense layers with ReLU activation functions, followed by
    Dropout layers. The decoder layers have a mirrored structure, starting from the latent space and expanding
    back to the original input dimension.

    Args:
        input_dim (int): The dimension of the input data.
        latent_dim (int): The dimension of the latent space.
        dropout_rate (float, optional): The dropout rate for the layers. Defaults to 0.1.
        latent_space_activation (str, optional): The activation function for the latent space layer. Defaults to 'tanh'.
        output_activation (str, optional): The activation function for the output layer. Defaults to 'tanh'.

    Returns:
        tuple: A tuple containing the autoencoder, encoder, and decoder models as TensorFlow Model objects:
            - autoencoder (Model): The complete autoencoder model.
            - encoder (Model): The encoder model.
            - decoder (Model): The decoder model.
    """

    # Calculate the number of layers for the encoder and decoder based on the input dimension
    num_layers = int(np.log2(input_dim / latent_dim))

    # Create the input layer
    input_layer = Input(shape=(input_dim,))

    # Build the encoder layers
    encoder = input_layer
    for i in range(num_layers):
        n_neurons = int(input_dim / (2 ** (i + 1)))
        encoder = Dense(n_neurons, activation='relu')(encoder)
        encoder = Dropout(dropout_rate)(encoder)

    # Latent space layer
    latent_space = Dense(latent_dim, activation=latent_space_activation, name='latent_space')(encoder)

    # Build the decoder layers
    decoder = latent_space
    for i in range(num_layers, 0, -1):
        n_neurons = int(input_dim / (2 ** i))
        decoder = Dense(n_neurons, activation='relu')(decoder)
        decoder = Dropout(dropout_rate)(decoder)

    # Output layer
    decoder_output = Dense(input_dim, activation=output_activation)(decoder)

    # Create the autoencoder, encoder, and decoder models
    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    encoder = Model(inputs=input_layer, outputs=latent_space)

    encoded_input = Input(shape=(latent_dim,))
    decoded_output = encoded_input
    decoder_layers = autoencoder.layers[-(num_layers * 2 + 1):]
    for layer in decoder_layers:
        decoded_output = layer(decoded_output)
    decoder = Model(inputs=encoded_input, outputs=decoded_output)

    return autoencoder, encoder, decoder


def create_autoencoder_enhanced(input_dim, latent_dim, dropout_rate=0.1,
                                latent_space_activation='tanh', output_activation='tanh', l1_reg=1e-5, l2_reg=1e-5):
    """
    Create an autoencoder with the given parameters. The autoencoder consists of an encoder and a decoder.
    The encoder compresses the input data into a lower-dimensional latent space, and the decoder reconstructs
    the original data from the latent space representation. The number of encoder and decoder layers and their
    neuron counts are determined by the input_dim and latent_dim parameters.

    The architecture of the autoencoder is dynamically constructed based on the input parameters. Specifically,
    the number of layers in the encoder and decoder is determined by the relationship between the input_dim
    and latent_dim. The encoder layers consist of Dense layers with ReLU activation functions, followed by
    Dropout layers. The decoder layers have a mirrored structure, starting from the latent space and expanding
    back to the original input dimension.

    Args:
        input_dim (int): The dimension of the input data.
        latent_dim (int): The dimension of the latent space.
        dropout_rate (float, optional): The dropout rate for the layers. Defaults to 0.1.
        latent_space_activation (str, optional): The activation function for the latent space layer. Defaults to 'tanh'.
        output_activation (str, optional): The activation function for the output layer. Defaults to 'tanh'.

    Returns:
        tuple: A tuple containing the autoencoder, encoder, and decoder models as TensorFlow Model objects:
            - autoencoder (Model): The complete autoencoder model.
            - encoder (Model): The encoder model.
            - decoder (Model): The decoder model.
    """

    # Calculate the number of layers for the encoder and decoder based on the input dimension
    num_layers = int(np.ceil(np.log2(input_dim / latent_dim)))

    # Create the input layer
    input_layer = Input(shape=(input_dim,))

    # Build the encoder layers
    encoder = input_layer
    n_neurons_list = []  # Store the number of neurons for each encoder layer
    for i in range(num_layers):
        n_neurons = int(input_dim / (2 ** (i + 1)))
        n_neurons_list.append(n_neurons)
        encoder = Dense(n_neurons, activation=None, kernel_regularizer=tf.keras.regularizers.l1_l2(l1_reg, l2_reg))(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = LeakyReLU()(encoder)
        encoder = Dropout(dropout_rate * (i + 1) / num_layers)(encoder)

    # Latent space layer
    latent_space = Dense(latent_dim, activation=latent_space_activation, name='latent_space')(encoder)

    # Build the decoder layers
    decoder = latent_space
    for i, n_neurons in enumerate(reversed(n_neurons_list)):
        decoder = Dense(n_neurons, activation=None, kernel_regularizer=tf.keras.regularizers.l1_l2(l1_reg, l2_reg))(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = LeakyReLU()(decoder)
        decoder = Dropout(dropout_rate * (num_layers - i) / num_layers)(decoder)

    # Output layer
    decoder_output = Dense(input_dim, activation=output_activation)(decoder)

    # Create the autoencoder, encoder, and decoder models
    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    encoder = Model(inputs=input_layer, outputs=latent_space)

    encoded_input = Input(shape=(latent_dim,))
    decoded_output = encoded_input
    decoder_layers = autoencoder.layers[-(num_layers * 4 + 1):]
    for layer in decoder_layers:
        decoded_output = layer(decoded_output)
    decoder = Model(inputs=encoded_input, outputs=decoded_output)

    return autoencoder, encoder, decoder


def fit_autoencoder(autoencoder_func, train_data, val_data, input_dim, latent_dim,
                    dropout_rate=0.01, epochs=1000, batch_size=32, patience=50,
                    optimizer=RMSprop, loss=tf.keras.losses.Huber(), monitor='val_loss', optimizer_kwargs=None):
    """
    Create and fit an autoencoder to the given training and validation data.

    Args:
        autoencoder_func (function): The create_autoencoder function to create the autoencoder architecture.
        train_data (np.array): The training data as a NumPy array.
        val_data (np.array): The validation data as a NumPy array.
        input_dim (int): The dimension of the input data.
        latent_dim (int): The dimension of the latent space.
        dropout_rate (float, optional): The dropout rate for the layers. Defaults to 0.01.
        epochs (int, optional): The number of epochs to train the autoencoder. Defaults to 1000.
        batch_size (int, optional): The batch size for training the autoencoder. Defaults to 32.
        patience (int, optional): The number of epochs with no improvement before stopping training. Defaults to 50.
        optimizer (Optimizer, optional): The optimizer to use for training the autoencoder. Defaults to RMSprop().
        loss (Loss, optional): The loss function to use for training the autoencoder. Defaults to tf.keras.losses.Huber().
        monitor (str, optional): The metric to monitor for early stopping. Defaults to 'val_loss'.
        optimizer_kwargs (dict, optional): A dictionary of keyword arguments to pass to the optimizer. Defaults to None.
    Returns:
        tuple: A tuple containing the fitted autoencoder, encoder, and decoder models as TensorFlow Model objects,
               and the training history:
            - autoencoder (Model): The complete autoencoder model.
            - encoder (Model): The encoder model.
            - decoder (Model): The decoder model.
            - history (History): The training history.
    """
    # Define autoencoder architecture
    autoencoder, encoder, decoder = autoencoder_func(input_dim, latent_dim, dropout_rate=dropout_rate)
    # Create a new optimizer instance
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer(**optimizer_kwargs)

    # Compile and train the autoencoder with early stopping
    autoencoder.compile(optimizer=optimizer, loss=loss)
    early_stopping = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    history = autoencoder.fit(train_data, train_data,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(val_data, val_data),
                              callbacks=[early_stopping])

    return autoencoder, encoder, decoder, history


def fit_denoising_autoencoder(autoencoder_func, x_train, y_train, x_val, y_val, input_dim, latent_dim,
                              dropout_rate=0.01, epochs=5000, batch_size=32, patience=100,
                              optimizer=RMSprop, loss=tf.keras.losses.Huber(), monitor='val_loss', optimizer_kwargs=None):
    """
    Create and fit a denoising autoencoder to the given noisy training and validation data.

    Args:
        autoencoder_func (function): The create_autoencoder function to create the autoencoder architecture.
        x_train (np.array): The noisy training input data as a NumPy array.
        y_train (np.array): The original (non-noisy) training output data as a NumPy array.
        x_val (np.array): The noisy validation input data as a NumPy array.
        y_val (np.array): The original (non-noisy) validation output data as a NumPy array.
        input_dim (int): The dimension of the input data.
        latent_dim (int): The dimension of the latent space.
        dropout_rate (float, optional): The dropout rate for the layers. Defaults to 0.01.
        epochs (int, optional): The number of epochs to train the denoising autoencoder. Defaults to 5000.
        batch_size (int, optional): The batch size for training the denoising autoencoder. Defaults to 32.
        patience (int, optional): The number of epochs with no improvement before stopping training. Defaults to 100.
        optimizer (Optimizer, optional): The optimizer to use for training the denoising autoencoder. Defaults to RMSprop().
        loss (Loss, optional): The loss function to use for training the denoising autoencoder. Defaults to tf.keras.losses.Huber().
        monitor (str, optional): The metric to monitor for early stopping. Defaults to 'val_loss'.
        optimizer_kwargs (dict, optional): A dictionary of keyword arguments to pass to the optimizer. Defaults to None.
    Returns:
        tuple: A tuple containing the fitted denoising autoencoder, encoder, and decoder models as TensorFlow Model objects,
               and the training history:
            - autoencoder (Model): The complete denoising autoencoder model.
            - encoder (Model): The encoder model.
            - decoder (Model): The decoder model.
            - history (History): The training history.
    """
    # Define autoencoder architecture
    autoencoder, encoder, decoder = autoencoder_func(input_dim, latent_dim, dropout_rate=dropout_rate)
    # Create a new optimizer instance
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer(**optimizer_kwargs)
    # Compile and train the autoencoder with early stopping
    autoencoder.compile(optimizer=optimizer, loss=loss)
    early_stopping = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    history = autoencoder.fit(x_train, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(x_val, y_val),
                              callbacks=[early_stopping])

    return autoencoder, encoder, decoder, history


def plot_history(history, title=None):
    """
    Plot the training and validation losses from the training history of an autoencoder.

    Args:
        history (History): The training history of the autoencoder.
        title (str, optional): The title of the plot. Defaults to None.
    Returns:
        None
    """
    # Plot the training and validation losses
    plt.style.use('default')
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Losses\n{title}')
    plt.show()
