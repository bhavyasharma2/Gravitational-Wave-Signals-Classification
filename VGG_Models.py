import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

# Load metadata
df_metadata = pd.read_csv("data/gravitational_waves_dataset/trainingset_v1d1_metadata.csv")
df_new = pd.read_csv("data/L1_O3b.csv")

num_labels = df_metadata["label"].nunique()
label_names = list(df_metadata["label"].value_counts().index)

# Load image datasets
train_dir = "data/gravitational_waves_dataset/train/train/"
(train_dataset, validation_dataset) = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(256, 256),
    subset="both",
    shuffle=True,
    seed=69,
    validation_split=0.2,
)

test_dir = "data/gravitational_waves_dataset/test/test/"
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=(256, 256),
    shuffle=False
)

class_names = train_dataset.class_names

# Prefetching
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Custom Label Smoothing Loss
class LabelSmoothingLoss(tf.keras.losses.Loss):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        num_classes = y_pred.shape[-1]
        confidence = 1.0 - self.smoothing
        smooth_labels = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
        smooth_labels = smooth_labels * confidence + (1 - confidence) / num_classes
        return tf.keras.losses.categorical_crossentropy(smooth_labels, y_pred)

# Model building blocks
def conv_block(input_tensor, filters, kernel_size):
    x = layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x

def inception_residual_block(input_tensor, filters):
    x1 = layers.Conv2D(filters, (1, 1), padding='same')(input_tensor)

    x2 = layers.Conv2D(filters, (1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x2 = layers.ReLU()(x2)
    x2 = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(x2)

    x3 = layers.Conv2D(filters, (1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x3 = layers.ReLU()(x3)
    x3 = layers.Conv2D(filters, (5, 5), padding='same', kernel_regularizer=regularizers.l2(0.01))(x3)

    x4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    x4 = layers.Conv2D(filters, (1, 1), padding='same', kernel_regularizer=regularizers.l2(0.01))(x4)

    return layers.concatenate([x1, x2, x3, x4], axis=-1)

def post_conv_block(input_tensor, filters, kernel_size):
    x = layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    return x

def branch(input_tensor):
    x = conv_block(input_tensor, filters=32, kernel_size=(3, 3))
    for _ in range(3):
        x = inception_residual_block(x, filters=32)
        x = post_conv_block(x, filters=32, kernel_size=(3, 3))
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    return x

# Input and base model
input_tensor = layers.Input(shape=(256, 256, 3))
base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model.trainable = False
output = base_model.output

# Attention-based multi-branch model
branches = [branch(output) for _ in range(4)]
x = layers.concatenate(branches)
intermediate_features = layers.Reshape((1, 1, -1))(x)
x = layers.Conv2D(filters=1, kernel_size=1, activation='relu')(intermediate_features)
x = layers.Flatten()(x)
x = layers.Activation('softmax')(x)
x = layers.Reshape((-1, 1, 1))(x)
x = layers.Multiply()([intermediate_features, x])
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(num_labels, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.005, restore_best_weights=True)
checkpoint_path = 'model_checkpoint.h5'
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

# Train model
history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset, callbacks=[checkpoint, early_stopping])

# Load best model
best_model = load_model(checkpoint_path)
base_model.trainable = True

fine_tune_from = 'block5_conv1'
for layer in base_model.layers:
    if layer.name == fine_tune_from:
        break
    layer.trainable = False

best_model.compile(optimizer=optimizers.Adam(learning_rate=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fine-tune
history_fine_tune = best_model.fit(train_dataset, epochs=10, validation_data=validation_dataset, callbacks=[checkpoint, early_stopping])

# Evaluate
test_loss, test_accuracy = best_model.evaluate(test_dataset)

# Predictions
predicted = best_model.predict(test_dataset)
predicted_labels = np.argmax(predicted, axis=1)

true_labels = []
for batch in test_dataset:
    true_labels.extend(batch[1].numpy())
true_labels = np.array(true_labels)

label_names.sort()
df_confusion = pd.DataFrame(0, index=label_names, columns=label_names)

for i in range(len(predicted_labels)):
    df_confusion.loc[label_names[predicted_labels[i]], label_names[true_labels[i]]] += 1

print(df_confusion)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(true_labels, predicted_labels))

cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='PRGn', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
