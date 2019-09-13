import numpy as np
from dataset import load, tf


def recuperacion(generators, image_file):
    """
    Genera una imagen de un paisaje recuperado luego de un incendio.
    El generador transformara la imagen de a trozos.
        Args:
        - generators: Una lista de modelos generadores que tomara una imagen y 
                     generara su versionrecuperada.
        - image_file: El directorio de la imagen que desea tranformar.
        
        Returns: 
            La imagen resultante almacenada en un `Tensor`
            de valores 0-1 de tipo float32.
    """
    img_prueba = load(image_file)
    img_prueba = tf.image.resize(img_prueba, (768, 1024))
    resultado = np.zeros((768, 1024, 3))
    x, y, _ = img_prueba.shape
    fila = 0
    columna = 0
    intervalo_y = (y - 256)//5
    intervalo_x = (x - 256)//3
    while 256 + fila < x:
        while 256 + columna < y:
            #graficar(img_prueba/255, img_prueba[: , 0+columna:256+columna]/255 )
            part = img_prueba[fila:256+fila , columna:256+columna]
            part = (part*2)-1
            part_gen = generar_imagen(generators, part[None, ...])
            part = resultado[fila : 256+fila , columna : 256+columna]
            part[part == 0] = part_gen[part == 0]
            part[:] = np.mean([part,part_gen],axis=0)
            columna += intervalo_y
        columna = 0
        fila += intervalo_x
    resultado = (resultado+1)/2
    resultado = tf.cast(resultado, tf.float32)
    return resultado

def generar_imagen(generators, image):
    for generator in generators:
        image = generator(image, training=True)
    image = tf.image.resize(image[0], [258,258], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = image[1:-1, 1:-1]
    return image.numpy()