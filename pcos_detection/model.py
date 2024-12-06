from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_and_train_model(X_train, y_train, X_val, y_val): 

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.5,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Build a more optimized model
    model = Sequential([

        # basic low-level features: edges, corners, and textures from the input image.
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        #complex features by combining the low-level features (shapes, patterns).
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        #higher-level features and pattern
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        #2d image to 1d image for dense layer
        Flatten(),
        Dense(128, activation='relu'), 
        Dropout(0.5), #50% neuron deactivated

        Dense(1, activation='sigmoid')  # Binary classification: two outcomes either pcos or not
    ])

    # Compile with Adam optimizer
    model.compile(optimizer=Adam(learning_rate=0.00003),  
                  loss='binary_crossentropy', #difference between the true labels and the predicted probabilities
                  metrics=['accuracy'])

    # Callbacks for learning rate reduction and early stopping
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Train the model with data augmentation
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                        epochs=30,  # Extended epochs for gradual improvement
                        validation_data=(X_val, y_val),
                        callbacks=[lr_scheduler])

    return model, history