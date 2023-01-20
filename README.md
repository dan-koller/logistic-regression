# Python Logistic Regression

This is a simple implementation of logistic regression in Python built from scratch. It uses the breast cancer dataset from sklearn. The dataset contains 569 samples and 30 features. The goal is to predict whether a tumor is malignant or benign.

## Requirements

-   Python 3
-   Packages from `requirements.txt`

## Installation

1. Clone the repository

```bash
git clone https://github.com/dan-koller/Python-Logistic-Regression
```

2. Create a virtual environment\*

```bash
python3 -m venv venv
```

3. Activate the virtual environment\*

```bash
source venv/bin/activate
```

4. Install the requirements\*

```bash
pip3 install -r requirements.txt
```

5. Run the app\*

```bash
python3 main.py
```

_\*) You might need to use python and pip instead of python3 and pip3 depending on your system._

## Usage

The app will run and print the accuracy of the model. The model is trained on 80% of the data and tested on the remaining 20%. The model is trained using gradient descent. The learning rate and number of iterations can be changed in `main.py`.

The app prints the mse, accuracy and loss for the training and testing data and compares the custom model to the sklearn model.

The output is plotted and saved to `data/graph.png` and looks like this:

![Graph plot](./data/graph.jpg "Plot of the data")

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
