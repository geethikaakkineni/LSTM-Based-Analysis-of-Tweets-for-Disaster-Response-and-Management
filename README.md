# Disaster Tweets Classification with LSTM

This project classifies tweets as disaster-related or not using an LSTM-based neural network built with TensorFlow/Keras.

## Project Structure

- `disastertweet.py`: Main script for preprocessing, model building, training, and evaluation.
- Jupyter notebook and images: Additional resources and diagrams.

## Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- scikit-learn

Install dependencies with:
```sh
pip install tensorflow numpy pandas scikit-learn
```

## Usage

Run the main script:
```sh
python disastertweet.py
```

The script will:
- Tokenize and pad example tweets
- Split data into training and test sets
- Build and train an LSTM model
- Evaluate and print the model's loss and accuracy

## Example

Sample output:
```
Loss: 0.6931, Accuracy: 50.0
```

## Notes

- The dataset in this example is very small and for demonstration only.
- For real applications, use a larger and more diverse
