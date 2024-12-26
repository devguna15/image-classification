import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(data_dir: str, model_path: str):
    # Data preparation
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Model definition
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(train_generator, validation_data=validation_generator, epochs=10)

    # Save the model
    model.save(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    train_model("data/images", "model/image_classifier.h5")
