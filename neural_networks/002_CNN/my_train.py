import matplotlib.pyplot as plt
import tensorflow as tf

from my_cnn_network import ImageClassification

# parameters
batch_size = 128
epochs = 20


def main():
    IC = ImageClassification()
    x_train, x_valid, x_test, y_train, y_valid, y_test = IC.create()
    model = IC.build()

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_valid, y_valid),
    )

    score = model.evaluate(x_test, y_test, verbose=0)

    print("Test loss for model:", score[0])
    print("Test accuracy for model:", score[1])

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
