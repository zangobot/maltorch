from secmlt.models.base_trainer import BaseTrainer
from secmlt.models.base_model import BaseModel
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def calculate_score(y_preds: np.array) -> float:
    num_benign = np.count_nonzero(y_preds == 0)
    num_malicious = np.count_nonzero(y_preds == 1)
    y_score = num_malicious / (num_benign + num_malicious)
    return y_score


class SmoothingClassifierTrainer(BaseTrainer):

    def __init__(self, num_epochs: int = 10, patience: int = 5, output_directory_path: str = None):
        self.num_epochs = num_epochs
        self.patience = patience
        self.output_directory_path = output_directory_path

        # Check if the directory exists
        if not os.path.exists(output_directory_path):
            # If the directory does not exist, create it
            os.makedirs(output_directory_path)
            print(f"Directory {output_directory_path} created.")
        else:
            print(f"Directory {output_directory_path} already exists.")

    def train(self, model: BaseModel, training_dataloader: DataLoader, validation_dataloader: DataLoader) -> BaseModel:
        """
        Train a model with the given dataloader.

        Parameters
        ----------
        model : BaseModel
            Model to train.
        dataloader : DataLoader
            Training dataloader.

        Returns
        -------
        BaseModel
            The trained model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters())

        validation_losses = []
        validation_accuracies = []
        best_loss = sys.maxsize
        best_epoch = 0

        for epoch in tqdm(range(self.num_epochs)):
            running_loss = 0.0
            train_correct = 0
            train_total = 0

            model.train()
            for inputs, labels in tqdm(training_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, labels.float())
                loss.backward()

                optimizer.step()
                running_loss += loss.item()

                y_preds = outputs.round()

                train_total += labels.size(0)
                train_correct += (y_preds == labels).sum().item()

            # Validation Set Eval
            model.eval()
            eval_train_correct = 0
            eval_train_total = 0
            running_loss = 0
            preds = []
            truths = []

            with torch.no_grad():
                for inputs, labels in tqdm(validation_dataloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    outputs = torch.squeeze(outputs)
                    y_preds = outputs.round()
                    y_preds = y_preds.detach().cpu().numpy().ravel()

                    new_labels = torch.full((y_preds.shape[0],), labels.item())
                    new_labels = new_labels.to(device)
                    loss = criterion(outputs, new_labels.float())

                    running_loss += loss.mean().item()

                    y_score = calculate_score(y_preds)
                    if y_score >= model.threshold:
                        y_pred = 1
                    else:
                        y_pred = 0

                    preds.extend([y_pred])
                    truths.extend(labels.detach().cpu().numpy().ravel())

                    try:
                        eval_train_total += labels.size(0)
                    except IndexError:
                        eval_train_total += 1
                    eval_train_correct += y_pred == labels.item()
            val_loss = running_loss / eval_train_total

            cm = confusion_matrix(np.array(truths).astype(int), np.array(preds).round().astype(int), normalize=None)
            cm_normalized = confusion_matrix(np.array(truths).astype(int), np.array(preds).round().astype(int),
                                             normalize="true")
            val_accuracy = accuracy_score(truths, preds)
            val_precision = precision_score(truths, preds)
            val_recall = recall_score(truths, preds)
            val_f1score = f1_score(truths, preds)

            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    os.path.join(self.output_directory_path, "model_state_dict.pth")
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion
                }, os.path.join(self.output_directory_path, "model.pth"))

                print("Validation accuracy: {}".format(val_accuracy))
                print("Validation F1 Score: {}".format(val_f1score))
                print("Validation precision: {}".format(val_precision))
                print("Validation recall: {}".format(val_recall))
                print("Validation Positive Rate: {}".format(sensitivity))
                print("Validation Negative Rate: {}".format(specificity))
                print("Validation loss: {}".format(val_loss))
                print("Confusion matrix:\n{}".format(cm))
                print("Normalized confusion matrix:\n{}".format(cm_normalized))

                with open(os.path.join(self.output_directory_path, "validation_set_metrics.out"), "w") as output_file:
                    output_file.write("Validation accuracy: {}\n".format(val_accuracy))
                    output_file.write("Validation F1 Score: {}\n".format(val_f1score))
                    output_file.write("Validation precision: {}\n".format(val_precision))
                    output_file.write("Validation recall: {}\n".format(val_recall))
                    output_file.write("Validation Positive Rate: {}\n".format(sensitivity))
                    output_file.write("Validation Negative Rate: {}\n".format(specificity))
                    output_file.write("Validation loss: {}\n".format(val_loss))
                    output_file.write("Confusion matrix:\n{}\n".format(cm))
                    output_file.write("Normalized confusion matrix:\n{}\n".format(cm_normalized))

            if epoch - best_epoch >= self.patience:
                print(
                    "The model hasn't improved for {} epochs. Stop training.\n Best loss: {}: Last epochs losses:{}".format(
                        self.patience, best_loss, validation_losses[-self.patience:]))
                break  # Stop training

            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy)
        return model








