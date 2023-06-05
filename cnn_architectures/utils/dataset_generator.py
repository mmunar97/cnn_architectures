import tensorflow as tf
import os
import warnings
from typing import Tuple

def get_dataset(image_dir: str,
                mask_dir: str,
                num_epochs: int,
                batch_size: int,
                img_size: Tuple[int, int, int] = (256, 256, 3),
                mask_size: Tuple[int, int, int] = (256, 256, 2),
                shuffle: bool = True,
                repeat: bool = True):


    # Listar los archivos en cada directorio
    images = os.listdir(image_dir)
    masks = os.listdir(mask_dir)

    # Ordenar los nombres de archivo alfabéticamente
    images.sort()
    masks.sort()

    # Crear una lista con los nombres de archivo en el orden deseado
    images_list = []
    masks_list = []
    for image, mask in zip(images, masks):
        if image == mask:
            images_list.append(os.path.join(image_dir, image))
            masks_list.append(os.path.join(mask_dir, mask))
        else:
            warnings.warn(f'Names do not match: {image} - {mask}')


    def load_and_preprocess_image(imgPath):
#         img_size = (256, 256, 3)
        img = tf.io.read_file(imgPath)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [img_size[0], img_size[1]])
        if img_size[2] == 1:
            img = tf.image.rgb_to_grayscale(img)
        img /= 255
        return img

    def load_and_preprocess_mask(maskPath):
#         mask_size = (256, 256, 2)
        mask = tf.io.read_file(maskPath)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, [mask_size[0], mask_size[1]])
        if mask_size[2] == 1:
            mask = tf.expand_dims(mask, axis=-1)
        elif mask_size[2] == 2:
            mask_aux = 255 - mask
            mask = tf.concat([mask, mask_aux], axis=-1)
        elif mask_size[2] == 28:
            # DoubleUNet output type
            mask = tf.concat([mask, mask], axis=-1)
        mask /= 255
        return mask

    # Crear un objeto tf.data.Dataset a partir de la lista de archivos
    image_dataset = tf.data.Dataset.from_tensor_slices(images_list).map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    mask_dataset = tf.data.Dataset.from_tensor_slices(masks_list).map(load_and_preprocess_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=250)
    dataset = dataset.batch(batch_size=batch_size)
    if repeat:
        dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset