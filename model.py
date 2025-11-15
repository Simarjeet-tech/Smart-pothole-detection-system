import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_model(img_size=(224, 224, 3), base_trainable=False, dropout_rate=0.3):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = base_trainable

    inputs = layers.Input(shape=img_size)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model
