# 🌭 Hotdog or Not Hotdog? 🍔

Welcome to the _Silicon Valley_ inspired dataset: **Hotdog or Not Hotdog**! This repository contains images of hotdogs and other foods, perfectly curated for all your machine learning shenanigans.

![Hotdog or Not Hotdog](https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExdzRlbDJmZ3phaWFscHpmdHN4dzdtbGZpN2RqY3BvOGt3ajVueXdzaSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26FmQcjUrHfNjKQGA/giphy.gif)

## Why Hotdog or Not Hotdog? 🤔

Inspired by the iconic Jian-Yang's app from the TV series _Silicon Valley_, this dataset is your gateway to mastering the subtle art of food recognition. Whether you're building the next revolutionary food identification app or just having fun with deep learning, we've got you covered.

## Dataset Contents 🍽️

The dataset is organized into two main folders:

- **hotdog/**: Pictures of the glorious, delicious hotdogs.
- **not_hotdog/**: Pictures of other foods (burgers, pizzas, sushi, you name it).

Here's a sneak peek:

```
train/
├── hotdog/
│ ├── hotdog1.jpg
│ ├── hotdog2.jpg
│ └── ...
└── not_hotdog/
├── not_hotdog1.jpg
├── not_hotdog2.jpg
└── ...
```

## Usage 🚀

1. **Clone the repo**:

   ```bash
   git clone https://github.com/truepositive/hotdog_nothotdog
   ```

2. **Train your model**: Use your favorite machine learning framework to train a classifier on the dataset. Here's an example using TensorFlow/Keras:

   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

   # Prepare data generators
   datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
   train_generator = datagen.flow_from_directory('data/', target_size=(150, 150), batch_size=32, class_mode='binary', subset='training')
   validation_generator = datagen.flow_from_directory('data/', target_size=(150, 150), batch_size=32, class_mode='binary', subset='validation')

   # Build the model
   model = Sequential([
       Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
       MaxPooling2D((2, 2)),
       Conv2D(64, (3, 3), activation='relu'),
       MaxPooling2D((2, 2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(1, activation='sigmoid')
   ])

   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   model.fit(train_generator, epochs=10, validation_data=validation_generator)
   ```

3. **Predict the hotdog-ness**:

   ```python
   from tensorflow.keras.preprocessing import image
   import numpy as np

   def is_hotdog(img_path):
       img = image.load_img(img_path, target_size=(150, 150))
       img_array = image.img_to_array(img) / 255.0
       img_array = np.expand_dims(img_array, axis=0)
       prediction = model.predict(img_array)
       return "Hotdog!" if prediction[0][0] > 0.5 else "Not Hotdog!"

   print(is_hotdog('path/to/your/test/image.jpg'))
   ```

## Contributing 🤝

Feel free to fork this repository, open issues, and submit pull requests. We welcome contributions from all food enthusiasts and tech geeks alike!

## License 📜

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🫶

- Thanks to HBO's "Silicon Valley" for the hilarious inspiration.
- Shoutout to all foodies and machine learning enthusiasts for making data science delicious!

---

_Disclaimer : No hotdogs were harmed in the making of this dataset._
