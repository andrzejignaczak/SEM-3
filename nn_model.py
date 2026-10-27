"""
nn_model.py
Moduł uczenia maszynowego (PyTorch): Architektura sieci MLP, normalizacja danych,
rekonstrukcja symetrycznej macierzy sztywności 6x6 oraz pętla treningowa.
"""

from typing import Tuple, List, Dict, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from tpms_generator import vector_21_to_matrix, matrix_to_vector_21


class GyroidStiffnessMLP(nn.Module):
    """
    Wielowarstwowy perceptron (MLP) przewidujący 21 niezależnych elementów
    górnego trójkąta symetrycznej macierzy sztywności 6x6 na podstawie [nx, ny, nz, t].
    """
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 21,
        hidden_dims: List[int] = [64, 128, 128, 64],
        activation: str = "gelu",
        dropout_rate: float = 0.0,
        use_norm: bool = True,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_norm:
                layers.append(nn.LayerNorm(h_dim))

            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "silu":
                layers.append(nn.SiLU())
            else:
                layers.append(nn.GELU())

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_matrix_6x6(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zwraca symetryczny tensor (batch_size, 6, 6) lub (6, 6).
        """
        out_21 = self.forward(x)
        if out_21.ndim == 1:
            mat = torch.zeros((6, 6), dtype=out_21.dtype, device=out_21.device)
            indices = torch.triu_indices(6, 6)
            mat[indices[0], indices[1]] = out_21
            mat = mat + mat.T - torch.diag(torch.diag(mat))
            return mat
        else:
            bs = out_21.shape[0]
            mat = torch.zeros((bs, 6, 6), dtype=out_21.dtype, device=out_21.device)
            indices = torch.triu_indices(6, 6)
            for i in range(bs):
                mat[i, indices[0], indices[1]] = out_21[i]
                mat[i] = mat[i] + mat[i].T - torch.diag(torch.diag(mat[i]))
            return mat


class StiffnessSurrogateModel:
    """
    Klasa-opakowanie do zarządzania modelem sieci, normalizacją i ewaluacją.
    """
    def __init__(
        self,
        hidden_dims: List[int] = [64, 128, 128, 64],
        lr: float = 0.003,
        activation: str = "gelu",
        dropout_rate: float = 0.0,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.hidden_dims = hidden_dims
        self.lr = lr
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.model = GyroidStiffnessMLP(
            input_dim=4,
            output_dim=21,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate,
            use_norm=True,
        ).to(self.device)

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_r2": [],
        }

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 120,
        batch_size: int = 64,
        val_size: float = 0.2,
        learning_rate: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int, float, float, float], None]] = None,
    ) -> Dict[str, List[float]]:
        """
        Wysokowydajna wektorowa pętla treningowa z walidacją, MSELoss, AdamW i R^2.
        """
        if learning_rate is not None:
            self.lr = learning_rate

        # Podział na train / validation
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X, y, test_size=val_size, random_state=42
        )

        # Dopasowanie scalerów
        X_train_norm = self.scaler_X.fit_transform(X_train_raw)
        X_val_norm = self.scaler_X.transform(X_val_raw)

        y_train_norm = self.scaler_y.fit_transform(y_train_raw)
        y_val_norm = self.scaler_y.transform(y_val_raw)

        # Konwersja na tensory na docelowym urządzeniu
        t_X_train = torch.tensor(X_train_norm, dtype=torch.float32, device=self.device)
        t_y_train = torch.tensor(y_train_norm, dtype=torch.float32, device=self.device)
        t_X_val = torch.tensor(X_val_norm, dtype=torch.float32, device=self.device)
        t_y_val = torch.tensor(y_val_norm, dtype=torch.float32, device=self.device)

        n_samples = t_X_train.shape[0]
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.6, patience=12, min_lr=1e-5
        )
        criterion = nn.MSELoss()

        self.history = {"train_loss": [], "val_loss": [], "val_r2": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            # Losowa permutacja indeksów dla mini-batchów
            perm = torch.randperm(n_samples, device=self.device)
            epoch_losses = []

            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                b_X = t_X_train[idx]
                b_y = t_y_train[idx]

                optimizer.zero_grad()
                preds = self.model(b_X)
                loss = criterion(preds, b_y)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_train_loss = float(np.mean(epoch_losses))

            # Walidacja
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(t_X_val)
                val_loss_tensor = criterion(val_preds, t_y_val)
                avg_val_loss = float(val_loss_tensor.item())

                # Obliczenie R² (ważone wariancją cech, aby szum na zerowych komponentach nie zaburzał metryki)
                val_preds_np = val_preds.cpu().numpy()
                val_true_np = t_y_val.cpu().numpy()
                val_r2 = float(r2_score(val_true_np, val_preds_np, multioutput="variance_weighted"))

            scheduler.step(avg_val_loss)

            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["val_r2"].append(val_r2)

            if progress_callback is not None:
                progress_callback(epoch, epochs, avg_train_loss, avg_val_loss, val_r2)

        self.is_trained = True
        return self.history

    def predict(self, X_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wykonuje inferencję:
        X_input: kształt (4,) lub (N, 4)
        Zwraca:
        - pred_vec_21: kształt (21,) lub (N, 21)
        - pred_matrix_6x6: kształt (6, 6) lub (N, 6, 6)
        """
        is_single = False
        if X_input.ndim == 1:
            is_single = True
            X_arr = X_input.reshape(1, -1)
        else:
            X_arr = X_input

        X_norm = self.scaler_X.transform(X_arr)
        self.model.eval()

        with torch.no_grad():
            tensor_X = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
            preds_norm = self.model(tensor_X).cpu().numpy()

        # Odwracanie normalizacji y
        preds_denorm = self.scaler_y.inverse_transform(preds_norm)

        # Rekonstrukcja symetrycznej macierzy 6x6
        mat_6x6 = vector_21_to_matrix(preds_denorm)

        if is_single:
            return preds_denorm[0], mat_6x6[0]
        return preds_denorm, mat_6x6

    def save_checkpoint(self, path: str):
        """Zapisuje wagi modelu i stan scalerów."""
        checkpoint = {
            "model_state": self.model.state_dict(),
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "history": self.history,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Wczytuje wagi modelu i stan scalerów."""
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        self.scaler_X = checkpoint["scaler_X"]
        self.scaler_y = checkpoint["scaler_y"]
        self.history = checkpoint.get("history", {})
        self.model.load_state_dict(checkpoint["model_state"])
        self.is_trained = True
