# src/models/mobilenet_model.py

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def build_mobilenet_model(input_shape, num_classes, train_base=False, fine_tune_at=None):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze or partially unfreeze base model
    if not train_base:
        base_model.trainable = False
    else:
        base_model.trainable = True
        if fine_tune_at is not None:
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    # ❗ we do NOT rescale here, because dataloader already does rescale=1./255
    x = base_model(inputs, training=False if not train_base else True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model
