# Handwriting Recognizer

A **machine learning-powered** character recognition tool that uses Convolutional Neural Networks (CNNs) to identify handwritten **digits** and **letters**. The project lets you draw characters on a Pygame-based canvas and watch the AI guess them in real time using pre-trained models.

This project supports:

* **Digit Recognition** using MNIST
* **Letter Recognition** using EMNIST
* **Combined Character Recognition** (digits + letters)

Built using **Keras** and **TensorFlow**, this project emphasizes key ML concepts and provides scripts for training, testing, and visualizing models.

---

## 📁 Project Structure

```bash
letter-checker/
├── README.md
├── assets/
│   └── fonts/
│       └── OpenSans-Regular.ttf
├── combined/
│   ├── models/
│   │   └── combined_model.keras
│   └── writing_combined.py
├── combined_detection.py        # Runs combined digit+letter recognition
├── detection.py                 # Runs digit-only recognition
├── requirements.txt
└── separate/
    ├── models/
    │   ├── digit_model.keras
    │   └── letter_model.keras
    ├── writing_digits.py        # For training/testing digit model
    └── writing_letters.py       # For training/testing letter model (not visualized yet)
```

---

## Features

* 🎨 Draw on a grid and let the AI recognize your input
* 🤖 Pre-trained CNN models using MNIST (digits) and EMNIST (letters)
* 🧠 Toggle between digit-only, letter-only, or combined recognition
* 📊 Model accuracies:

  * **Digits only:** 99.4%
  * **Letters only:** 97.5%
  * **Combined model:** 96.7%

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Handwriting-Recognizer.git
cd letter-checker
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 🔀 Combined Mode (Digits + Letters)

This will use the combined model trained on both MNIST and EMNIST.

```bash
python combined_detection.py combined/models/combined_model.keras
```

You’ll be presented with a Pygame grid where you can draw either a digit or a letter. The AI will guess which character it is.

---

### 🔢 Digits-Only Mode

To test just digit recognition (trained only on MNIST):

```bash
python detection.py separate/models/digit_model.keras
```

Draw a digit (0–9) and get instant predictions.

---

### 🔤 Letters-Only Mode

The letter-only model is trained and available at:

```
separate/models/letter_model.keras
```

However, **the interactive visualization (like detection.py) is not implemented yet** for this mode. You can still train and test the model using:

```bash
python writing_letters.py
```

---

## Training Your Own Models

Scripts for training your own CNNs from scratch are provided:

* Train digits model:

  ```bash
  python writing_digits.py
  ```
* Train letters model:

  ```bash
  python writing_letters.py
  ```
* Train combined model:

  ```bash
  python combined/writing_combined.py
  ```

Each script saves its model to the respective `models/` directory.

---

## 🛠️ Built With

* [Python](https://www.python.org/)
* [Keras](https://keras.io/)
* [TensorFlow](https://www.tensorflow.org/)
* [Pygame](https://www.pygame.org/)
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)

---

## 📌 Notes

* The system currently distinguishes **uppercase letters only** from EMNIST.
* Be mindful of drawing clarity; especially for characters like `0`, `O`, `I`, `1`, `L`, etc.
* The Pygame canvas clears automatically after each guess.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## Contributing

Pull requests are welcome! If you’d like to help improve letter recognition or add lowercase support, feel free to contribute.

---
