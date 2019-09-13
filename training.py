import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

from loss_opt import discriminator_loss, generator_loss
from recuperacion import recuperacion

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


@tf.function()
def train_step(input_image, target_image, generator, discriminator):
    """
    Este es un paso de entrenamiento en el que se calculará el error de la
    imagen generada con respecto a la imagen deseada y actualizara los 
    pesos de las redes neuronales.
        Args:
        - input_image: La imagen de entrada en bacth = 1, shape = (1, 256, 256, 3)
        - target_image: La imagen que se quiere obtener en bacth = 1, shape = (1, 256, 256, 3)
        - generator: El modelo generador a entrenar.
        - discriminator: El modelo discriminador para el entrenamiento.
        
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:

        generated_image = generator(input_image, training=True)

        discriminated_gen_image = discriminator([generated_image, input_image], training=True)

        discriminated_target_image = discriminator([target_image, input_image], training=True)

        discr_loss = discriminator_loss(discriminated_target_image, discriminated_gen_image)

        gen_loss = generator_loss(discriminated_gen_image, generated_image, target_image)

        generator_gradients = gen_tape.gradient(
            gen_loss,
            generator.trainable_variables
        )
        discriminator_gradients = discr_tape.gradient(
            discr_loss,
            discriminator.trainable_variables
        )
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

def train(dataset, epochs, generator, discriminator, checkpoint_dir = 'training_checkpoints'):
    """
    Inicia el entrenamiento de toda la arquitectura.
        Args:
        - dataset: Un generador de tensorflow que cargara los datos de entrenamiento.
        - epochs: Es el numero de repeticiones que entrenara sobre el dataset.
        - generator: El modelo generador a entrenar.
        - discriminator: El modelo discriminador para el entrenamiento.
        - checkpoint_dir: El directorio donde se gardara el modelo como punto de control.
    """
    load_models(generator, discriminator, checkpoint_dir) # carga los modelos
    
    for epoch in range(epochs):
        start = time.time()
        for input_image, target_image in dataset:
            train_step(input_image, target_image, generator, discriminator)    
        if (epoch+1) % 20 == 0:
            generator.save(checkpoint_dir + f'/{generator.name}.h5')
            discriminator.save(checkpoint_dir + f'/{discriminator.name}.h5')
            print(f"Punto de control guardado para el paso {str(epoch + 1)}")

        print (f'Tiempo necesario para la época: {str(epoch + 1)} es {time.time()-start} sec')

def load_models(generator, discriminator, checkpoint_dir="training_checkpoints"):
    if os.path.exists(checkpoint_dir):
        if os.path.exists(checkpoint_dir+f"/{generator.name}.h5"):
            generator.load_weights(checkpoint_dir + f'/{generator.name}.h5')
            print(f'Se cargo el {generator.name}.')
        else:
            print(f'El {generator.name} es nuevo')
        if os.path.exists(checkpoint_dir+f"/{discriminator.name}.h5"):
            discriminator.load_weights(checkpoint_dir + f'/{discriminator.name}.h5')
            print(f'Se cargo el {discriminator.name}.')
        else:
            print(f'El {discriminator.name} es nuevo')
    else:
        os.mkdir(checkpoint_dir)
        print("Iniciando entrenamiento desde cero.")
        
def generate_images(generator, disc, test_input, tar, checkpoint_dir='training_checkpoints'):
    # the training=True is intentional here since
    # we want the batch statistics while running the generator
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    start = time.time()
    prediction = generator(test_input, training=True)
    print(time.time()-start)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        img = (display_list[i] + 1) / 2
        plt.imshow(img)
        plt.axis('off')
    plt.show()

