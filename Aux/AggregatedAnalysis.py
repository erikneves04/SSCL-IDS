import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch

from Scarf_modified_diff_pospairs import SCARF
from utils import get_embeddings_labels

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class ExampleDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (array-like): As características de entrada.
            labels (array-like): As etiquetas de saída.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {'feature': self.features[idx], 'label': self.labels[idx]}
        return sample


def load_model(model_path):
    """Carregar o modelo a partir do caminho especificado."""
    return joblib.load(model_path)

def load_data(dataset_path):
    """Carregar dados do dataset CSV especificado."""
    df = pd.read_csv(dataset_path)
    # Excluir colunas irrelevantes (exemplo: ID, Timestamp, etc.), mantendo as colunas de features
    df = df.drop(columns=['Timestamp', 'ID'])  # Ajuste conforme necessário
    x_data = df.iloc[:, :-1]  # Colunas de Features
    y_data = df["Label"]  # Coluna Target
    return x_data, y_data

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plotar a matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    tick_marks = np.arange(len(set(y_true)))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(set(y_true))
    ax.set_yticklabels(set(y_true))
    plt.colorbar(ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Carregar dados de teste
    test_data_path = "datasets/CICIDS-2017/dataset.csv"  # Altere conforme necessário
    x_test, y_test = load_data(test_data_path)

    # Carregar os modelos salvos
    self_model_path = 'saves/supervised/scarf_model.pkl'  # Ajuste o caminho conforme necessário
    supervised_model_path = 'saves/supervised/rf_model.pkl'  # Ajuste o caminho conforme necessário

    # Carregar o modelo self (SCARF)
    ckpt = torch.load('checkpoints/scarf_checkpoint.pth')  # Caminho do checkpoint do modelo SCARF
    train_args = ckpt["args"]
    model = SCARF(
        input_dim=train_args.input_dim,
        emb_dim=train_args.embedding_dim,
        corruption_rate=train_args.corruption_rate,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Obter embeddings de teste (model SCARF)
    test_ds = ExampleDataset(x_test.to_numpy(), y_test.to_numpy(), columns=x_test.columns)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, drop_last=False)
    test_embeddings, _ = get_embeddings_labels(model, test_loader, 'cpu', to_numpy=False, normalize=True)

    # Carregar o modelo supervisionado
    rf_model = load_model(supervised_model_path)

    # Previsões para o modelo SCARF (self)
    self_predictions = rf_model.predict(test_embeddings)

    # Previsões para o modelo supervisionado (RandomForest ou outro)
    supervised_model = load_model(supervised_model_path)
    supervised_predictions = supervised_model.predict(x_test)

    # Relatórios de classificação para o modelo self e supervised
    print("### Classification Report - Self Model (SCARF embeddings) ###")
    print(classification_report(y_test, self_predictions))

    print("### Classification Report - Supervised Model (Raw Data) ###")
    print(classification_report(y_test, supervised_predictions))

    # Plotar as matrizes de confusão
    plot_confusion_matrix(y_test, self_predictions, title="Confusion Matrix - Self Model (SCARF)")
    plot_confusion_matrix(y_test, supervised_predictions, title="Confusion Matrix - Supervised Model")
