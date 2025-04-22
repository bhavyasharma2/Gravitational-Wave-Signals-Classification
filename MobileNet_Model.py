import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, regularizers

from sklearn.metrics import classification_report, confusion_matrix

# Set paths
TRAIN_DIR = "data/gravitational_waves_dataset/train"
TEST_DIR = "data/gravitational_waves_dataset/test"
METADATA_CSV = "data/gravitational_waves_dataset/trainingset_v1d1_metadata.csv"

class LabelSmoothingLoss(tf.keras.losses.Loss):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        num_classes = y_pred.shape[-1]
        confidence = 1.0 - self.smoothing
        smooth_labels = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
        smooth_labels = smooth_labels * confidence + (1 - confidence) / num_classes
        return tf.keras.losses.categorical_crossentropy(smooth_labels, y_pred)

def load_datasets():
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        image_size=(256, 256),
        batch_size=32,
        validation_split=0.2,
        subset="both",
        seed=69
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        image_size=(256, 256),
        shuffle=False
    )
    return train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE), \
           val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE), \
           test_ds

def build_model(input_shape=(256, 256, 3), num_classes=22):
    def conv_block(x, filters, kernel_size):
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        return x

    def inception_residual_block(x, filters):
        x1 = layers.Conv2D(filters, (1, 1), padding='same')(x)

        x2 = layers.Conv2D(filters, (1, 1), padding='same')(x)
        x2 = layers.ReLU()(x2)
        x2 = layers.Conv2D(filters, (3, 3), padding='same')(x2)

        x3 = layers.Conv2D(filters, (1, 1), padding='same')(x)
        x3 = layers.ReLU()(x3)
        x3 = layers.Conv2D(filters, (5, 5), padding='same')(x3)

        x4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        x4 = layers.Conv2D(filters, (1, 1), padding='same')(x4)

        return layers.concatenate([x1, x2, x3, x4], axis=-1)

    def post_conv_block(x, filters, kernel_size):
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        return x

    def branch(x):
        x = conv_block(x, 32, (3, 3))
        for _ in range(3):
            x = inception_residual_block(x, 32)
            x = post_conv_block(x, 32, (3, 3))
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Dropout(0.5)(x)
        return x

    inputs = layers.Input(shape=input_shape)
    base_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False

    x = base_model.output
    branches = [branch(x) for _ in range(4)]
    x = layers.concatenate(branches)
    output = layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=output)

def main():
    # Load metadata
    df = pd.read_csv(METADATA_CSV)
    label_names = sorted(df['label'].unique().tolist())

    train_ds, val_ds, test_ds = load_datasets()
    model = build_model(num_classes=len(label_names))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(train_ds, validation_data=val_ds, epochs=3, callbacks=[early_stop])

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc:.4f}")

    predictions = model.predict(test_ds)
    predicted_labels = np.argmax(predictions, axis=1)

    true_labels = np.concatenate([y.numpy() for _, y in test_ds])
    print(classification_report(true_labels, predicted_labels, target_names=label_names))

    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PRGn', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
