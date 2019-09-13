import os
import sys
import numpy as np
import tensorflow as tf
from dataset import load, load_image_train, load_image_test
from pix2pix_models import Discriminator, Generator
from training import train_step, train, generator_optimizer, discriminator_optimizer, generate_images, load_models
#tf.debugging.set_log_device_placement(True)


PATH = os.path.abspath('./')
PATH_IN = PATH + '/input_images_'
PATH_OUT = PATH + '/output_images_'

BUFFER_SIZE = 400
BATCH_SIZE = 1

def load_dataset(sufijo='a'):
    file_names = [sufijo+'/'+arch.name for arch in os.scandir(PATH_IN+sufijo) if arch.is_file()]
    if file_names:
        n = len(file_names)
        train_n = round(n * 0.40)
        
        train_urls = file_names[:train_n] # datos de entrenamiento.
        np.random.shuffle(train_urls)
        
        test_urls = file_names[train_n:] # datos de prueba.
        np.random.shuffle(test_urls)

        train_dataset = tf.data.Dataset.from_tensor_slices(train_urls)
        train_dataset = train_dataset.map(load_image_train)
        train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE)
        train_dataset = train_dataset.batch(BATCH_SIZE)

        test_dataset = tf.data.Dataset.from_tensor_slices(test_urls)
        test_dataset = test_dataset.map(load_image_test)
        test_dataset = test_dataset.batch(BATCH_SIZE)
        
        return train_dataset, test_dataset


def entrenar(dataset, epochs, sufijo='a'):
    generator = Generator("generator_"+sufijo)
    discriminator = Discriminator("discriminator_"+sufijo)
    train(dataset, epochs, generator, discriminator)
    
def testear(data_test, cantidad = 10, sufijo='a'):
    generator = Generator("generator_"+sufijo)
    discriminator = Discriminator("discriminator_"+sufijo)
    load_models(generator, discriminator, checkpoint_dir="saved_model")
    contador = 0
    for inp_img, re_img in data_test:
        generate_images(generator, discriminator, inp_img, re_img)
        contador += 1
        if contador >= cantidad:
            break
        

if __name__ == "__main__":
    try:
        sufijo = sys.argv[1:][0]
        if sufijo == 'a' or sufijo == 'b':
            train_dataset, test_dataset = load_dataset(sufijo)
            #entrenar(train_dataset, 10, sufijo)
            testear(test_dataset, sufijo=sufijo)
    except:
        print("Error, falta argumento a/b")
    
