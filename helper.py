import os
import tensorflow as tf

def make_segmentation_dataset(
    image_dir,
    mask_dir,
    batch_size=20,
    target_size=(256, 256),
    image_color_mode="grayscale",
    shuffle=True,
    seed=1,
    binarize_mask=True,
):
    def list_images(folder):
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        files = [os.path.join(folder, f)
                 for f in os.listdir(folder)
                 if f.lower().endswith(exts)]
        return sorted(files)

    img_paths = list_images(image_dir)
    msk_paths = list_images(mask_dir)

    if len(img_paths) == 0 or len(msk_paths) == 0:
        raise ValueError(f"Found {len(img_paths)} images and {len(msk_paths)} masks.")
    if len(img_paths) != len(msk_paths):
        raise ValueError(f"Image and mask count mismatch: {len(img_paths)} vs {len(msk_paths)}")

    channels = 1 if image_color_mode == "grayscale" else 3

    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(img_paths), 2048), seed=seed, reshuffle_each_iteration=True)

    def load_pair(img_path, mask_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=channels, expand_animations=False)
        img = tf.image.resize(img, target_size, method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
        mask = tf.image.resize(mask, target_size, method="nearest")
        mask = tf.cast(mask, tf.float32) / 255.0

        if binarize_mask:
            mask = tf.where(mask > 0.5, 1.0, 0.0)

        return img, mask

    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
    
    
def make_test_dataset(test_dir, target_size=(256,256), channels=1, batch_size=1):
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(exts)])

    ds = tf.data.Dataset.from_tensor_slices(paths)

    def load_img(p):
        x = tf.io.read_file(p)
        x = tf.image.decode_image(x, channels=channels, expand_animations=False)
        x = tf.image.resize(x, target_size, method="bilinear")
        x = tf.cast(x, tf.float32) / 255.0
        return x

    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, paths


