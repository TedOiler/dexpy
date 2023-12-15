import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.backend import clear_session
import gc

from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from matplotlib import pyplot as plt

from cordex_discrete import cordex_discrete
from cordex_continuous import cordex_continuous

from latent_bo import objective_function


# def objective_function_tf(X, m, n, J_cb=None, noise=0):
#     batch_size = tf.shape(X)[0]
#     ones = tf.ones((batch_size, m, 1))
#     X = tf.reshape(X, (-1, m, n))
#     Z = tf.concat([ones, tf.matmul(X, J_cb)], axis=2)
#
#     try:
#         M = tf.linalg.inv(tf.matmul(Z, Z, transpose_a=True))
#     except tf.errors.InvalidArgumentError:
#         return tf.constant(1e10)
#
#     result = tf.linalg.trace(M) + tf.random.normal([], mean=0, stddev=noise)
#     result = tf.where(result < 0, tf.constant(1e10), result)
#     return tf.reduce_mean(result)

# HELPERS--------------------------------------
def create_train_val_set_random(runs, n_x, scalars, optimality, J_cb,
                                num_designs=1000, test_size=0.2, random_state=42, noise=None, max_iterations=100000,
                                epsilon=1e-10, min=-1, max=1):
    design_matrices = []
    valid_count = 0
    for _ in tqdm(range(max_iterations)):
        if valid_count >= num_designs:
            break

        candidate_matrix = np.random.uniform(min, max, size=(runs, n_x[0]))

        Z = np.hstack([np.ones((runs, 1)), candidate_matrix @ J_cb])
        ZTZ = Z.T @ Z
        determinant = np.linalg.det(ZTZ)

        if determinant > epsilon:
            design_matrices.append(candidate_matrix)
            valid_count += 1

    if valid_count < num_designs:
        print(f"Warning: Only {valid_count} valid design matrices found after {max_iterations} iterations")

    design_matrices = np.stack(design_matrices)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_designs = scaler.fit_transform(design_matrices.reshape(num_designs, -1))

    if noise is not None:
        noisy_designs = normalized_designs + noise * np.random.normal(size=normalized_designs.shape)
        noisy_designs = np.clip(noisy_designs, -1, 1)

        x_train, x_val, y_train, y_val = train_test_split(noisy_designs, normalized_designs, test_size=test_size,
                                                          random_state=random_state)
        return x_train, x_val, y_train, y_val
    else:
        train_data, val_data = train_test_split(normalized_designs, test_size=test_size, random_state=random_state)
        return train_data, val_data


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
        if cordex == cordex_discrete:
            opt_design, opt_cr = cordex_discrete(runs=runs, f_list=n_x, scalars=scalars, levels=[-1, 1], epochs=1,
                                                 optimality=optimality, J_cb=J_cb, disable_bar=True)
        elif cordex == cordex_continuous:
            opt_design, opt_cr = cordex_continuous(runs=runs, f_list=n_x, scalars=scalars, optimality=optimality,
                                                   J_cb=J_cb, epochs=1, final_pass_iter=1, main_bar=False,
                                                   final_bar=False)
        else:
            opt_design, opt_cr = None, None
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


def objective_function_tf(X, m, n, J_cb=None, noise=0):
    batch_size = tf.shape(X)[0]
    ones = tf.ones((batch_size, m, 1))
    X = tf.reshape(X, (-1, m, n))
    Z = tf.concat([ones, tf.matmul(X, J_cb)], axis=2)

    Z_transpose_Z = tf.matmul(Z, Z, transpose_a=True)
    det_Z_transpose_Z = tf.linalg.det(Z_transpose_Z)
    epsilon = 1e-06
    condition = tf.abs(det_Z_transpose_Z)[:, None, None] < epsilon

    identity_matrix = tf.eye(tf.shape(Z_transpose_Z)[1], tf.shape(Z_transpose_Z)[2])
    diagonal_part = tf.linalg.diag_part(Z_transpose_Z) + epsilon
    Z_transpose_Z_epsilon = Z_transpose_Z + tf.linalg.diag(diagonal_part - tf.linalg.diag_part(Z_transpose_Z))
    regularized_matrix = tf.where(condition, Z_transpose_Z_epsilon, Z_transpose_Z)

    M = tf.linalg.inv(regularized_matrix)
    result = tf.linalg.trace(M) + tf.random.normal([], mean=0, stddev=noise)
    result = tf.where(result < 0, tf.constant(1e10), result)
    return tf.reduce_mean(result)


def combined_loss(alpha, loss_function, m, n, J_cb=None, noise=0):
    def custom_loss(y_true, y_pred):
        reconstruction_loss = loss_function(y_true, y_pred)
        objective_value = objective_function_tf(y_pred, m, n, J_cb=J_cb, noise=noise)
        return (1 - alpha) * reconstruction_loss + alpha * objective_value

    return custom_loss


# ARCHITECTURE----------------------------------

def create_autoencoder(input_dim, latent_dim,
                       latent_space_activation='tanh', output_activation='tanh',
                       max_layers=None, alpha=0.0, base=2):
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
    num_layers = int(np.log(input_dim / latent_dim) / np.log(base))

    if max_layers is not None:
        num_layers = min(num_layers, max_layers)

    # Create the input layer
    input_layer = Input(shape=(input_dim,))

    # Build the encoder layers
    encoder = input_layer
    for i in range(num_layers):
        n_neurons = int(input_dim / (2 ** (i + 1)))
        encoder = Dense(n_neurons, activation=LeakyReLU(alpha=alpha))(encoder)

    # Latent space layer
    latent_space = Dense(latent_dim, activation=latent_space_activation, name='latent_space')(encoder)

    # Build the decoder layers
    decoder = latent_space
    for i in range(num_layers, 0, -1):
        n_neurons = int(input_dim / (2 ** i))
        decoder = Dense(n_neurons, activation=LeakyReLU(alpha=alpha))(decoder)

    # Output layer
    decoder_output = Dense(input_dim, activation=output_activation)(decoder)

    # Create the autoencoder, encoder, and decoder models
    autoencoder = Model(inputs=input_layer, outputs=decoder_output)
    encoder = Model(inputs=input_layer, outputs=latent_space)

    encoded_input = Input(shape=(latent_dim,))
    decoded_output = encoded_input
    decoder_layers = autoencoder.layers[-(num_layers + 1):]
    for layer in decoder_layers:
        decoded_output = layer(decoded_output)
    decoder = Model(inputs=encoded_input, outputs=decoded_output)

    return autoencoder, encoder, decoder


def create_autoencoder_enhanced(input_dim, latent_dim, dropout_rate=0.1,
                                latent_space_activation='tanh', output_activation='tanh', l1_reg=1e-5, l2_reg=1e-5,
                                max_layers=None, base=2):
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
    num_layers = int(np.ceil(np.log(input_dim / latent_dim) / np.log(base)))

    # Create the input layer
    input_layer = Input(shape=(input_dim,))

    # Build the encoder layers
    encoder = input_layer
    n_neurons_list = []  # Store the number of neurons for each encoder layer
    for i in range(num_layers):
        n_neurons = int(input_dim / (2 ** (i + 1)))
        n_neurons_list.append(n_neurons)
        encoder = Dense(n_neurons, activation=None, kernel_regularizer=tf.keras.regularizers.l1_l2(l1_reg, l2_reg))(
            encoder)
        encoder = BatchNormalization()(encoder)
        encoder = LeakyReLU()(encoder)
        encoder = Dropout(dropout_rate * (i + 1) / num_layers)(encoder)

    # Latent space layer
    latent_space = Dense(latent_dim, activation=latent_space_activation, name='latent_space')(encoder)

    # Build the decoder layers
    decoder = latent_space
    for i, n_neurons in enumerate(reversed(n_neurons_list)):
        decoder = Dense(n_neurons, activation=None, kernel_regularizer=tf.keras.regularizers.l1_l2(l1_reg, l2_reg))(
            decoder)
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


def create_vae(input_dim, latent_dim, dropout_rate=0.1,
               latent_space_activation='tanh', output_activation='tanh', l1_reg=1e-5, l2_reg=1e-5):
    num_layers = int(np.ceil(np.log2(input_dim / latent_dim)))

    input_layer = Input(shape=(input_dim,))

    # Build the encoder layers
    encoder = input_layer
    n_neurons_list = []  # Store the number of neurons for each encoder layer
    for i in range(num_layers):
        n_neurons = int(input_dim / (2 ** (i + 1)))
        n_neurons_list.append(n_neurons)
        encoder = Dense(n_neurons, activation=None, kernel_regularizer=tf.keras.regularizers.l1_l2(l1_reg, l2_reg))(
            encoder)
        encoder = BatchNormalization()(encoder)
        encoder = LeakyReLU()(encoder)
        encoder = Dropout(dropout_rate * (i + 1) / num_layers)(encoder)

    # Latent space layer
    z_mean = Dense(latent_dim, activation=latent_space_activation, name='z_mean')(encoder)
    z_log_var = Dense(latent_dim, activation=latent_space_activation, name='z_log_var')(encoder)

    # Reparameterization trick
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Instantiate the encoder model
    encoder = Model(input_layer, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    decoder = latent_inputs
    for i, n_neurons in enumerate(reversed(n_neurons_list)):
        decoder = Dense(n_neurons, activation=None, kernel_regularizer=tf.keras.regularizers.l1_l2(l1_reg, l2_reg))(
            decoder)
        decoder = BatchNormalization()(decoder)
        decoder = LeakyReLU()(decoder)
        decoder = Dropout(dropout_rate * (num_layers - i) / num_layers)(decoder)

    # Output layer
    decoder_output = Dense(input_dim, activation=output_activation)(decoder)

    # Instantiate the decoder model
    decoder = Model(latent_inputs, decoder_output, name='decoder')

    # VAE
    decoded_output = decoder(encoder(input_layer)[2])
    vae = Model(input_layer, decoded_output, name='vae')

    # Loss function
    reconstruction_loss = tf.keras.losses.Huber()(input_layer, decoded_output)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae, encoder, decoder


# TRAINING-------------------------------------
def fit_autoencoder_custom(autoencoder, encoder, decoder, train_data, val_data, epochs=1000, batch_size=32, patience=50,
                           optimizer=RMSprop, loss=tf.keras.losses.Huber(), monitor='val_loss',
                           alpha=1.0, m=None, n=None, J_cb=None, noise=0, optimizer_kwargs=None,
                           SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Create a new optimizer instance
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer(**optimizer_kwargs)
    # Create the custom combined loss function
    custom_loss = combined_loss(alpha, loss, m, n, J_cb=J_cb, noise=noise)

    # Compile and train the autoencoder with early stopping
    autoencoder.compile(optimizer=optimizer, loss=custom_loss)
    early_stopping = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    history = autoencoder.fit(train_data, train_data,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(val_data, val_data),
                              callbacks=[early_stopping])

    return autoencoder, encoder, decoder, history


def fit_autoencoder(autoencoder_func, train_data, val_data, input_dim, latent_dim,
                    dropout_rate=0.01, epochs=1000, batch_size=32, patience=50,
                    optimizer=RMSprop, loss=tf.keras.losses.Huber(), monitor='val_loss', optimizer_kwargs=None,
                    max_layers=None):
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
    autoencoder, encoder, decoder = autoencoder_func(input_dim, latent_dim, dropout_rate=dropout_rate,
                                                     max_layers=max_layers)
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
                              optimizer=RMSprop, loss=tf.keras.losses.Huber(), monitor='val_loss',
                              optimizer_kwargs=None, alpha=1.0, m=None, n=None, J_cb=None, noise=0, max_layers=None):
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
    autoencoder, encoder, decoder = autoencoder_func(input_dim, latent_dim, dropout_rate=dropout_rate,
                                                     max_layers=max_layers)
    # Create a new optimizer instance
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer(**optimizer_kwargs)
    # Compile and train the autoencoder with early stopping

    # Create the custom combined loss function
    custom_loss = combined_loss(alpha, loss, m, n, J_cb=J_cb, noise=noise)

    autoencoder.compile(optimizer=optimizer, loss=custom_loss)
    early_stopping = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    history = autoencoder.fit(x_train, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(x_val, y_val),
                              callbacks=[early_stopping])

    return autoencoder, encoder, decoder, history


def fit_vae(vae_func, train_data, val_data, input_dim, latent_dim,
            dropout_rate=0.01, epochs=1000, batch_size=32, patience=50,
            optimizer=RMSprop, loss=tf.keras.losses.MeanSquaredError(), monitor='val_loss', optimizer_kwargs=None):
    """
    Create and fit a variational autoencoder to the given training and validation data.

    Args:
        vae_func (function): The create_vae function to create the VAE architecture.
        train_data (np.array): The training data as a NumPy array.
        val_data (np.array): The validation data as a NumPy array.
        input_dim (int): The dimension of the input data.
        latent_dim (int): The dimension of the latent space.
        dropout_rate (float, optional): The dropout rate for the layers. Defaults to 0.01.
        epochs (int, optional): The number of epochs to train the VAE. Defaults to 1000.
        batch_size (int, optional): The batch size for training the VAE. Defaults to 32.
        patience (int, optional): The number of epochs with no improvement before stopping training. Defaults to 50.
        optimizer (Optimizer, optional): The optimizer to use for training the VAE. Defaults to RMSprop().
        loss (Loss, optional): The loss function to use for training the VAE. Defaults to tf.keras.losses.MeanSquaredError().
        monitor (str, optional): The metric to monitor for early stopping. Defaults to 'val_loss'.
        optimizer_kwargs (dict, optional): A dictionary of keyword arguments to pass to the optimizer. Defaults to None.
    Returns:
        tuple: A tuple containing the fitted VAE, encoder, and decoder models as TensorFlow Model objects,
               and the training history:
            - vae (Model): The complete VAE model.
            - encoder (Model): The encoder model.
            - decoder (Model): The decoder model.
            - history (History): The training history.
    """
    # Define VAE architecture
    vae, encoder, _ = vae_func(input_dim, latent_dim, dropout_rate=dropout_rate)

    # Create the decoder model
    decoder_input = Input(shape=(latent_dim,))
    decoder_output = vae.layers[-1](decoder_input)
    for layer in reversed(vae.layers[:-1]):
        if 'decoder' in layer.name:
            decoder_output = layer(decoder_output)
    decoder = Model(decoder_input, decoder_output)

    # Create a new optimizer instance
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = optimizer(**optimizer_kwargs)

    # Compile and train the VAE with early stopping
    vae.compile(optimizer=optimizer, loss=loss)
    early_stopping = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    history = vae.fit(train_data, train_data,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(val_data, val_data),
                      callbacks=[early_stopping])

    return vae, encoder, decoder, history


# PLOTTING-------------------------------------
def plot_history(history, title=None, threshold=None, margin=0.1, style='grayscale'):
    """
    Plot the training and validation losses from the training history of an autoencoder.

    Args:
        history (History): The training history of the autoencoder.
        title (str, optional): The title of the plot. Defaults to None.
    Returns:
        None
    """
    # Plot the training and validation losses
    if style == 'grayscale':
        # plt.style.use('grayscale')
        plt.plot(history.history['loss'], label='Training Loss', linestyle='-', marker='o', color='black')
        plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--', marker='x', color='black')
    else:
        plt.style.use('default')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Losses\n{title}')
    if threshold is not None:
        plt.ylim(top=threshold)

    min_loss = min(min(history.history['loss']), min(history.history['val_loss']))
    plt.ylim(bottom=max(min_loss - margin, 0))
    plt.show()


# HYPERPARAMETER TUNING------------------------

def hyperparameter_tuning(des_pure_train, des_pure_val, run, nx, J_cb, batch_size, optimizer_parameters, nn_epochs=100,
                          n_calls=10,
                          random_state=0, verbose=True, n_jobs=-1, n_random_starts=5, acq_func='EI',
                          acq_optimizer='sampling'):
    def latent_dim_objective(params):
        latent_dim = int(params[0])  # Extract and convert to integer
        exponent = params[1]  # Extract exponent

        if exponent == 0:
            alpha = 0
        else:
            alpha = 10.0 ** exponent  # Convert to order of magnitude

        # Create and train the autoencoder with the given latent_dim and alpha
        autoencoder, encoder, decoder = create_autoencoder(input_dim=des_pure_train.shape[1],
                                                           latent_dim=latent_dim,
                                                           max_layers=500,
                                                           alpha=alpha)

        _, _, _, history = fit_autoencoder_custom(autoencoder, encoder, decoder,
                                                  optimizer=RMSprop,
                                                  optimizer_kwargs=optimizer_parameters,
                                                  train_data=des_pure_train, val_data=des_pure_val,
                                                  patience=int(0.1 * nn_epochs), epochs=nn_epochs,
                                                  batch_size=batch_size,
                                                  alpha=1, m=run, n=sum(nx), J_cb=J_cb, noise=0)

        # Return the final validation loss as the objective to minimize
        val_loss = history.history['val_loss'][-1]
        del autoencoder
        del encoder
        del decoder
        gc.collect()
        return val_loss

    # Define the search space for latent_dim and alpha
    search_space = [(1, 15),  # Range for latent_dim
                    (-6, 0)]  # Range for exponent of alpha

    res = gp_minimize(latent_dim_objective, search_space, n_calls=n_calls, random_state=random_state, verbose=verbose,
                      n_jobs=n_jobs, n_random_starts=n_random_starts, acq_func=acq_func, acq_optimizer=acq_optimizer)

    # Extract the best latent_dim and alpha
    best_latent_dim = int(res.x[0])
    best_exponent = res.x[1]

    if best_exponent == 0:
        best_alpha = 0
    else:
        best_alpha = 10.0 ** best_exponent

    clear_session()
    return best_latent_dim, best_alpha


# PREDICTION-----------------------------------

def optimize_latent_variables(best_latent_dim, decoder, run, nx, J_cb, n_calls=80, random_state=0, verbose=True,
                              n_jobs=-1, n_random_starts=8, acq_func='EI', acq_optimizer='sampling'):
    """
    Optimize the latent variables using Gaussian Process.

    Parameters:
    - best_latent_dim: Optimal latent dimension from previous tuning
    - objective: Objective function to minimize
    - decoder: Decoder model from the autoencoder
    - run: Number of runs
    - nx: List of design variables
    - n_calls, random_state, verbose, n_jobs, n_random_starts, acq_func, acq_optimizer: Parameters for gp_minimize

    Returns:
    - optimal_latent_var: Optimal latent variables
    - optimal_cr: Optimal objective function value
    - optimal_des: Decoded design corresponding to the optimal latent variables
    """

    def objective(latent_var):
        latent_var = np.array(latent_var).reshape(1, -1)
        decoded = decoder.predict(latent_var)
        y_true = objective_function(decoded, m=run, n=sum(nx), J_cb=J_cb, noise=0)
        return y_true

    dimensions = [(-1., 1.) for _ in range(best_latent_dim)]
    res = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=random_state, verbose=verbose, n_jobs=n_jobs,
                      n_random_starts=n_random_starts, acq_func=acq_func, acq_optimizer=acq_optimizer)

    optimal_latent_var = res.x
    optimal_cr = res.fun
    optimal_des = decoder.predict(np.array(optimal_latent_var).reshape(1, -1)).reshape(run, sum(nx))
    search_history = res.x_iters
    eval_history = res.func_vals

    return optimal_latent_var, optimal_cr, optimal_des, search_history, eval_history
