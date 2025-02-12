import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Parameters
img_size = 224  # Input image size for DenseNet121
batch_size = 32
num_classes = 4  
epochs = 20
train_dir = r"ehr\images\train"
val_dir = r'ehr\images\val' 

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# Model Setup
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x) # Dense layer
predictions = Dense(num_classes, activation='softmax')(x) # Output layer

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(
    'best_model 1.keras',  
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint, early_stop]
)

# Fine-tuning: Unfreeze some base model layers and train further
for layer in base_model.layers[-50:]:  # Unfreeze last 50 layers
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue Training
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[checkpoint, early_stop]
)
# Print final accuracy
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

final_train_acc_fine = history_fine.history['accuracy'][-1]
final_val_acc_fine = history_fine.history['val_accuracy'][-1]

print(f"Final Training Accuracy (Before Fine-Tuning): {final_train_acc:.4f}")
print(f"Final Validation Accuracy (Before Fine-Tuning): {final_val_acc:.4f}")
print(f"Final Training Accuracy (After Fine-Tuning): {final_train_acc_fine:.4f}")
print(f"Final Validation Accuracy (After Fine-Tuning): {final_val_acc_fine:.4f}")

# Function to compute precision, recall, and F1-score
def compute_classification_metrics(generator, model):
    y_true = []
    y_pred = []

    for batch in generator:
        images, labels = batch
        preds = model.predict(images)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

        if len(y_true) >= generator.samples:
            break  # Stop when all samples are processed

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    return precision, recall, f1

# Compute metrics for validation set after fine-tuning
precision, recall, f1 = compute_classification_metrics(val_generator, model)

# Print Final Metrics
print("\n=== Final Training Metrics (Before Fine-Tuning) ===")
print(f"Loss: {final_train_loss:.4f}, Accuracy: {final_train_acc:.4f}")
print(f"Validation Loss: {final_val_loss:.4f}, Validation Accuracy: {final_val_acc:.4f}")

print("\n=== Final Training Metrics (After Fine-Tuning) ===")
print(f"Loss: {final_train_loss_fine:.4f}, Accuracy: {final_train_acc_fine:.4f}")
print(f"Validation Loss: {final_val_loss_fine:.4f}, Validation Accuracy: {final_val_acc_fine:.4f}")

print("\n=== Additional Metrics After Fine-Tuning (Validation Set) ===")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


# Save Final Model
model.save('final_model.keras')

# Optionally Save in .h5 Format
model.save('final_model.h5') 