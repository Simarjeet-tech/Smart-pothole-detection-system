import argparse
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_prep import get_generators
from visualize import plot_confusion_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model = load_model(args.model_path)
    _, _, test_gen = get_generators(args.data_dir, args.img_size, args.batch_size)

    preds = model.predict(test_gen)
    y_pred = (preds >= 0.5).astype(int).flatten()
    y_true = test_gen.classes

    labels = list(test_gen.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels, normalize=True, out_path="confusion_matrix.png")

if __name__ == "__main__":
    main()
