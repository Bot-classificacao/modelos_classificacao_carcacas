# %%
# libs necessarias
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# %%
# carregando as imagens
data_base_path = r'C:\Users\User\Downloads\Imagens'

# %%
# definido o tamanho do lote, largura e altura das imagens
batch_size = 32
img_width, img_height = 180, 180
epochs = 20
learning_rate = 0.0001  # Taxa de aprendizagem para o otimizador
print(img_height, img_width)

# %%
# separando o conjunto de dados em treinamento e validação
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_base_path,
    shuffle=True,
    seed=123,
    image_size=(img_height, img_width),
    subset="training",
    batch_size=batch_size,
    validation_split=0.2

)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_base_path,
    shuffle=True,
    seed=123,
    image_size=(img_height, img_width),
    subset="validation",
    batch_size=batch_size,
    validation_split=0.2,
)


# %%
nomes_classes = train_ds.class_names
nomes_classes

# %%
num_classe = len(nomes_classes)
num_classe


# %%
# Configurações automáticas de desempenho
AUTOTUNE = tf.data.AUTOTUNE

# Preparação dos dados de treinamento
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# Preparação dos dados de validação
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# %%
# Normalização das imagens
shape = (img_width, img_height, 3)
shape

# %%
# Normaliza os valores dos pixels das imagens para o intervalo [0,1]
normalizador = layers.Rescaling(1./255)

# %%
normalized_ds = train_ds.map(lambda x, y: (normalizador(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

# %%
# Criando o modelo de rede neural
model = Sequential([
    layers.Rescaling(1./255, input_shape=shape),  # Camada de normalização
    # Camada convolucional
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),  # Camada de max pooling
    # Camada convolucional
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),  # Camada de max pooling
    # Camada convolucional
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),  # Camada de max pooling
    # layers.Dropout(0.2),
    layers.Flatten(),  # Camada de flatten para transformar a saída em um vetor unidimensional
    # Camada densa (totalmente conectada)
    layers.Dense(128, activation='relu'),
    # Camada de saída com ativação sigmoid para classificação
    layers.Dense(num_classe, activation='softmax')
])

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])


# %%
model.summary()

# %%

# Treinando o modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# %%


def plotar_grafico():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    faixa_epochs = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(faixa_epochs, acc, label='Treino Accuracy')
    plt.plot(faixa_epochs, val_acc, label='Validacao Accuracy')

    plt.legend(loc='lower right')
    plt.title('Treino and Validacao Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(faixa_epochs, loss, label='Treino Loss')
    plt.plot(faixa_epochs, val_loss, label='Validacao Loss')
    plt.legend(loc='upper right')
    plt.title('Treino and Validacao Loss')
    plt.show()


# %%
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=shape),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# %%
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=shape),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classe, activation='softmax')
])

# %%

# Compilando o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])


# %%
model.summary()

# %%

# Treinando o modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# %%


def classificar_for_wpp(path_img):
    img = tf.keras.utils.load_img(
        path_img, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return f"classificada como {nomes_classes[np.argmax(score)]} \nCom uma accuracy de {
        100 * np.max(score):.2f} %."


# %%
classificar_for_wpp(
    r'C:\Users\User\Desktop\Robo_classificacao_whatsapp-main\resources\image_base\escasso.jpeg')


# %%
