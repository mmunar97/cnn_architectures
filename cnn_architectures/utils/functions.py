import tensorflow as tf
import numpy as np

def set_seed(seed: int = 28) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set as {seed}")