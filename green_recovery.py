import os
import sys
import time

from matplotlib import pyplot as plt
from pix2pix_models import Generator, tf
from recuperacion import recuperacion
from dataset import load

def main(generators, image_file):
    input_image = load(image_file)
    input_image = tf.image.resize(input_image, (768, 1024))
    gen_image = recuperacion(generators, image_file)
    
    plt.subplot(1,2,1); plt.imshow(input_image)
    plt.subplot(1,2,2); plt.imshow(gen_image)
    plt.show()
    
    if not os.path.exists('imagenes_generadas'):
        os.mkdir('imagenes_generadas') # Creamos el directorio si es que no existe.

    gen_image = tf.cast(gen_image * 255, tf.uint8)
    gen_image = tf.image.encode_jpeg(gen_image)
    tf.io.write_file(f'imagenes_generadas/imagen_generada_{str(time.time())}.jpg', gen_image)

if __name__ == "__main__":
    args = sys.argv[1:] # recibimos los argumentos de consola
    if args:
        image_file = args[0] # seleccionamos el primer argumento.
        if os.path.exists(image_file):
            if image_file[-4:] == '.jpg': # verifica si el archivo es una imagen.
                # cargamos los modelos.
                generators = [Generator("generator_a"), Generator("generator_b")]
                if os.path.exists('saved_models/generator_a.h5') and os.path.exists('saved_models/generator_b.h5'):
                    generators[0].load_weights('saved_models/generator_a.h5')
                    generators[1].load_weights('saved_models/generator_b.h5')
                    print("Modelos cargados....")
                # Ejecutamos la funcion principal
                main(generators, image_file)
            else:
                print("Este archivo no es una imagens JPG.")
        else:
            print("Este archivo no existe.")