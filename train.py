import os
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from model import build_model
from data_prep import get_generators

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--model_dir", type=str, default="models")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    train_gen, val_gen, _ = get_generators(
        args.data_dir, args.img_size, args.batch_size
    )

    model = build_model(img_size=(args.img_size, args.img_size, 3))

    checkpoint = ModelCheckpoint(
        os.path.join(args.model_dir, "best_model.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    early = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )

    csv_logger = CSVLogger(os.path.join(args.model_dir, "training_log.csv"))

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[checkpoint, early, reduce_lr, csv_logger]
    )

    model.save(os.path.join(args.model_dir, "final_model.h5"))
    print("Training complete!")

if __name__ == "__main__":
    main()
