import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def create_data_generators():
    """
    Create data generators for training and validation with data augmentation
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,                    # Normalize pixel values to [0,1]
        rotation_range=20,                    # Random rotation up to 20 degrees
        width_shift_range=0.2,               # Random width shifts
        height_shift_range=0.2,              # Random height shifts
        zoom_range=0.2,                       # Random zoom
        horizontal_flip=True,                   # Random horizontal flip
        brightness_range=[0.8, 1.2],         # Random brightness adjustment
        fill_mode='nearest'                     # Fill mode for transformations
    )
    
    # Validation data generator (only rescaling, no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        'dataset/test',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator

def build_model(num_classes):
    """
    Build VGG19 transfer learning model for dog breed classification
    """
    # Load VGG19 pre-trained on ImageNet
    base_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3)
    )
    
    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the complete model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    print("Model Architecture:")
    model.summary()
    
    return model

def train_model():
    """
    Train the VGG19 transfer learning model
    """
    print("Starting Dog Breed Classification Training...")
    
    # Create data generators
    train_generator, validation_generator = create_data_generators()
    
    # Save class indices
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(validation_generator.class_indices, f)
    print("Class indices saved to 'class_indices.json'")

    # Get number of classes dynamically
    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes.")

    # Build model
    model = build_model(num_classes)
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        'dog_breed_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Calculate steps
    num_train_samples = len(train_generator.filenames)
    num_val_samples = len(validation_generator.filenames)
    batch_size = train_generator.batch_size
    
    if num_train_samples == 0 or num_val_samples == 0:
        print("Error: No images found. Please run organize_dataset.py first and ensure images are present.")
        return None, None

    steps_per_epoch = num_train_samples // batch_size
    validation_steps = num_val_samples // batch_size
    
    print(f"Training samples: {num_train_samples}")
    print(f"Validation samples: {num_val_samples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=6,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(
        validation_generator,
        steps=validation_steps
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    model.save('dog_breed_model_final.h5')
    print("Model saved as 'dog_breed_model_final.h5'")
    
    return model, history

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Save plots
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training history plots saved as 'training_history.png'")

def evaluate_model(model):
    """
    Detailed model evaluation with classification report
    """
    # Create validation generator
    val_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = val_datagen.flow_from_directory(
        'dataset/test',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get predictions
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    
    # Get true labels
    y_true = validation_generator.classes
    
    # Get class labels
    class_labels = list(validation_generator.class_indices.keys())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_labels)

def plot_confusion_matrix(y_true, y_pred, class_labels):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Confusion matrix saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    # Train and evaluate the model
    model, history = train_model()
    evaluate_model(model)
