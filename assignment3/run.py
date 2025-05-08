import numpy as np
import matplotlib.pyplot as plt
import argparse
from MLP import MLP
from normalizer import LinearNormalizer


parser = argparse.ArgumentParser()
parser.add_argument("--infer", action="store_true", help="Inference mode")
parser.add_argument("-checkpoint", default='best', help="Training mode")
args = parser.parse_args()

def infer(normalizer, test_data, model=None):
    if model is None:
        # Load the best model
        checkpoint_path = args.checkpoint + '.npy'
        checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
        model = checkpoint["model"]
        print(f"Loaded model from {checkpoint_path}")
        
    # Inference
    y_pred = model.loss(test_data[:, :-1])
    y_pred = y_pred.reshape(-1)
    y_pred = normalizer.denormalize(np.hstack((test_data[:, :-1], y_pred.reshape(-1, 1))))[:, -1]
    data_denorm = normalizer.denormalize(test_data)
    # data_denorm = test_data.copy()
    y_true = data_denorm[:, -1]
    data = np.hstack((data_denorm, y_pred.reshape(-1, 1)))
    np.savetxt("test_data_with_predictions.csv", data, delimiter=",", header="housing_age,homeowner_income,house_price,pred_price", comments="", fmt="%d")

    y_pred_rate = np.abs(y_true - y_pred) / y_true
    success_5 = np.sum(y_pred_rate < 0.05) / len(y_pred_rate)
    success_10 = np.sum(y_pred_rate < 0.1) / len(y_pred_rate)
    success_20 = np.sum(y_pred_rate < 0.2) / len(y_pred_rate)
    success_50 = np.sum(y_pred_rate < 0.5) / len(y_pred_rate)
    print(f"Success rate (5%): {success_5 * 100:.2f}%, "
          f"Success rate (10%): {success_10 * 100:.2f}%, "
          f"Success rate (20%): {success_20 * 100:.2f}%, "
          f"Success rate (50%): {success_50 * 100:.2f}%")
    
    # RMSE and MAE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # plot the predictions with axis of x1 and x2 at 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_denorm[:, 0], data_denorm[:, 1], y_true, c='b', label="true_price", alpha=0.5)
    ax.scatter(data_denorm[:, 0], data_denorm[:, 1], y_pred, c='r', label="pred_price")
    ax.set_xlabel("housing_age")
    ax.set_ylabel("homeowner_income")
    ax.set_zlabel("house_price")
    ax.set_title("Predictions on Test Data")
    plt.legend()
    # 更改视角
    # ax.view_init(elev=20, azim=30)
    plt.show()
    # plt.savefig("predictions_plot.png")


def train_mlp():
    np.random.seed(42)

    data_path = 'MLP_data.csv'
    data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    data = data[:, 2:]
    data = np.random.permutation(data)
    num_train = int(0.7 * len(data))
    num_val = int(0.15 * len(data))
    num_test = len(data) - num_train - num_val

    model = MLP(
        input_dim=2,
        hidden_dims=[64, 128],
        output_dim=1,
        weight_scale=0.01,
        reg=0,
    )

    # Set up the normalizer
    normalizer = LinearNormalizer(data)
    data = normalizer.normalize(data)

    train_data = data[:num_train]
    val_data = data[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]

    if args.infer:
        infer(normalizer, test_data)
        return

    # Training parameters
    num_epochs = 200
    batch_size = 32
    learning_rate = 1e-4
    checkpoint_every = 50
    lr_decay_epoch = 50
    lr_decay = 0.05
    num_batches = int(np.ceil(num_train / batch_size))
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    for epoch in range(1, num_epochs+1):
        np.random.shuffle(train_data)
        for i in range(num_batches):
            x = train_data[i * batch_size:(i + 1) * batch_size, :-1]
            y = train_data[i * batch_size:(i + 1) * batch_size, -1].reshape(-1, 1)

            # y = normalizer.normalize(y)

            # Compute loss and gradients
            loss = model.loss(x, y)

            model.step(learning_rate)

        # Learning rate decay
        if epoch % lr_decay_epoch == 0:
            learning_rate *= (1 - lr_decay)

        # Compute training and validation loss
        train_loss = model.loss(train_data[:, :-1], train_data[:, -1].reshape(-1, 1))
        val_loss = model.loss(val_data[:, :-1], val_data[:, -1].reshape(-1, 1))
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # Check for best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_epoch = epoch

        if epoch % checkpoint_every == 0:
            save_dict = {
                "model": model,
                "epoch": epoch,
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history,
            }
            np.save(f"{epoch}.npy", save_dict)
            print(f"Checkpoint saved at epoch {epoch}")

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Learning Rate: {learning_rate:e}")

    # Save the best model
    save_dict = {
        "model": best_model,
        "epoch": best_epoch,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }
    np.save("best.npy", save_dict)
    print(f"Best model saved at epoch {best_epoch} with validation loss {best_val_loss} at best.npy")

    # plot the training and validation loss
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()

    infer(normalizer, test_data, model=best_model)

if __name__ == "__main__":
    train_mlp()