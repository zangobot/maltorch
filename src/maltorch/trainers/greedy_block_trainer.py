"""
Keane Lucas, Weiran Lin, Lujo Bauer, Michael K. Reiter, Mahmood Sharif
Training Robust ML-based Raw-Binary Malware Detectors
in Hours, not Months
ACM CCS 2024
https://dl.acm.org/doi/pdf/10.1145/3658644.3690208
"""
import torch.nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from dataclasses import dataclass
from typing import Optional
import torch.nn.functional as F

@dataclass
class GreedyBlockConfig:
    """Configuration for GreedyBlock augmentation."""
    budget: float = 0.03           # fraction of bytes to perturb (0.0–1.0)
    max_blocks: int = 5            # num_blocks is sampled in [1, max_blocks]
    num_iterations: int = 10       # gradient steps on the block embeddings
    alpha: float = 100.0             # step size in embedding space
    ig_steps: int = 8              # steps for (approximate) Integrated Gradients
    apply_prob: float = 1.0        # probability of applying GreedyBlock to a batch
    # if True, only augment malicious samples (label==1); else both classes
    only_malicious: bool = False
    padding_idx: int = 256 # Define padding index


class GreedyBlockAugmentor:
    """
    Implements GreedyBlock augmentation in embedding space.

    ASSUMPTIONS ABOUT `model`:
      - model has a method `embed(x: LongTensor[B, L]) -> FloatTensor[B, L, D]`
        that returns the byte embeddings.
      - model has a method `_forward_embed_x(e: FloatTensor[B, L, D])`
        that runs the rest of the network and returns logits of shape [B].
      - model is a binary classifier trained with BCEWithLogitsLoss.
    """

    def __init__(self, config: GreedyBlockConfig, loss_fn: Optional[torch.nn.Module] = None):
        self.config = config
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.BCEWithLogitsLoss()

    def augment_batch(self, model: torch.nn.Module, x: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
        """
        Apply GreedyBlock to (at most) one example in the batch.

        x: LongTensor [B, L] with byte indices.
        y: Tensor [B] with labels 0/1.
        Returns a new tensor x_aug of same shape.
        """
        if self.config.budget <= 0.0 or torch.rand(1).item() > self.config.apply_prob:
            return x

        B, L = x.shape
        device = x.device

        # choose candidate indices according to label filter
        candidate_indices = torch.arange(B, device=device)
        if self.config.only_malicious:
            candidate_indices = candidate_indices[y.view(-1) == 1]

        if candidate_indices.numel() == 0:
            return x

        idx = candidate_indices[torch.randint(0, candidate_indices.numel(), (1,))].item()
        #print("Candidate idx: ", idx)

        x_aug = x.clone()
        single_x = x_aug[idx]  # [L]
        single_y = y[idx].view(1)

        with torch.no_grad():
            aug_sample = self._greedyblock_single(model, single_x, single_y)
        x_aug[idx] = aug_sample
        return x_aug

    # ---------- core GreedyBlock logic ----------

    def _greedyblock_single(self, model: torch.nn.Module,
                            x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply GreedyBlock to a single sample.

        x: LongTensor [L]
        y: Tensor [1] (0/1)
        returns augmented x_adv: LongTensor [L]
        """
        L = int((x != self.config.padding_idx).sum().item())
        budget = self.config.budget
        num_bytes_to_perturb = int(budget * L)
        if num_bytes_to_perturb <= 0:
            return x

        # sample number of blocks between 1 and max_blocks
        num_blocks = torch.randint(1, self.config.max_blocks + 1, (1,)).item()
        num_blocks = max(1, min(num_blocks, num_bytes_to_perturb))
        block_size = max(1, num_bytes_to_perturb // num_blocks)

        # 1) compute attribution scores (approximate Integrated Gradients)
        model_was_training = model.training
        model.eval()
        attributions = self._integrated_gradients(model, x.unsqueeze(0), y)  # [1, L]
        attributions = attributions.squeeze(0)[:L].abs()  # [true_len]
        #attributions = attributions.abs().squeeze(0)  # [L]

        # 2) choose top-N block indices (contiguous blocks with highest attribution sum)
        block_indices = self._choose_top_blocks(attributions, block_size, num_blocks)  # [K]
        if block_indices.numel() == 0:
            if model_was_training:
                model.train()
            return x

        x_adv = x.clone()

        # 3) random initialization at chosen indices
        num_embeddings = model.embedding_layer().weight.shape[0]
        allowed = torch.arange(0, 256, device=x.device)
        rand_vals = allowed[torch.randint(0, allowed.numel(), (block_indices.numel(),),
                                  device=x.device, dtype=x_adv.dtype)]
        x_adv[block_indices] = rand_vals

        # ---- inner loop: align objective with IG (maximize flip objective) ----
        direction = (2 * y.float() - 1.0).to(x.device)
        # maximize scalar = -direction * logit -> pushes toward opposite clas

        # 4) iterative gradient-based update in embedding space
        for iteration in range(self.config.num_iterations):
            #print("iteration:", iteration)
            with torch.enable_grad():
                prev_block_values = copy.deepcopy(x_adv[block_indices])

                model.zero_grad(set_to_none=True)

                # e_full: [1, D, L] because your embed() transposes to [B, D, L]
                e_full = model.embed(x_adv.unsqueeze(0)).detach()
                e_full.requires_grad_(True)

                logits = model.forward_from_embeddings(e_full).view(-1)  # [1]
                target = y.float().to(logits.device)

                # Maximize change of direction
                #scalar = (-direction * logits).sum()
                #scalar.backward()

                # Maximize loss
                loss = self.loss_fn(logits, target)
                loss.backward()

                grads = e_full.grad.detach()[0]  # [D, L]

                # --- slice along L, then transpose to [K, D] ---
                g_block = grads[:, block_indices]  # [D, K]
                g_block = g_block.t().contiguous()  # [K, D]

                #print("||g_block|| mean:", g_block.norm(dim=1).mean().item())

                e_block = e_full.detach()[0, :, block_indices]  # [D, K]
                e_block = e_block.t().contiguous()  # [K, D]

                # gradient step in embedding space
                e_block_perturbed = e_block + self.config.alpha * g_block  # [K, D]

                #print("Block perturbed: ", e_block_perturbed)
                # project to nearest token embeddings
                emb_weight = model.embedding_layer().weight.detach()  # [V, D]
                emb_allowed = emb_weight[0:256]  # [256, D]
                new_byte_vals = self._closest_embedding_indices(emb_allowed, e_block_perturbed)

                # Old: without restricting padding embedding
                #new_byte_vals = self._closest_embedding_indices(
                #    emb_weight, e_block_perturbed
                #)  # [K]

                # write back to token sequence (x_adv: [L])
                x_adv[block_indices] = new_byte_vals
                after_block_values = copy.deepcopy(new_byte_vals)
                #print("Diff: ", torch.sum(torch.abs(prev_block_values-after_block_values)))

            # optional early stopping
            if hasattr(model, "threshold"):
                with torch.no_grad():
                    logits = model(x_adv.unsqueeze(0)).view(-1)
                    probs = torch.sigmoid(logits)
                    pred = (probs >= model.threshold).float()
                    if pred.item() != y.item():
                        break

        if model_was_training:
            model.train()
        return x_adv

    # ---------- helpers ----------

    def _integrated_gradients(self, model: torch.nn.Module,
                              x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: LongTensor [1, L]
        returns attributions: [1, L]
        """
        steps = self.config.ig_steps
        device = x.device

        with torch.enable_grad():
            baseline = torch.full_like(x, self.config.padding_idx) # baseline = torch.zeros_like(x) # if 0 padding

            # e_start, e_end: [1, D, L]
            e_start = model.embed(baseline).detach()
            e_end = model.embed(x).detach()

            total_grad = torch.zeros_like(e_end, device=device)  # [1, D, L]
            direction = (2 * y.float() - 1.0).to(device)  # {-1,+1}

            for alpha in torch.linspace(0, 1, steps, device=device):
                e_interp = ((1 - alpha) * e_start + alpha * e_end).detach()  # [1, D, L]
                e_interp.requires_grad_(True)

                model.zero_grad(set_to_none=True)
                logits = model.forward_from_embeddings(e_interp).view(-1)  # [1]

                # Maximize this to push toward opposite class
                #scalar = (-direction * logits).sum()
                #grad = torch.autograd.grad(scalar, e_interp, retain_graph=False, create_graph=False)[0]
                #total_grad += grad.detach()

                # Maximize loss
                target = y.float().to(logits.device)
                loss = self.loss_fn(logits, target)
                loss.backward()
                total_grad += e_interp.grad.detach()  # [1, D, L]

            ig = (e_end - e_start) * total_grad / steps  # [1, D, L]

            # reduce over embedding dim D to get per-position attribution over L
            #attributions = ig.norm(p=1, dim=1) # [1, L] Old
            attributions = ig.abs().sum(dim=1) # [1, L] (L1 over D)
            # optional: zero out PAD positions so they never look "important"
            mask = (x != self.config.padding_idx).float()  # [1, L]
            attributions = attributions * mask

        return attributions

    def _choose_top_blocks(self, attributions: torch.Tensor,
                           block_size: int, num_blocks: int) -> torch.Tensor:
        """
        Choose non-overlapping contiguous blocks with highest attribution sum.
        attributions: [L]
        returns concatenated indices of all blocks: [num_blocks * block_size] (or fewer).
        """
        L = attributions.shape[0]
        if block_size >= L:
            return torch.arange(L, device=attributions.device)

        # sliding-window sums via conv1d
        kernel = torch.ones(1, 1, block_size, device=attributions.device)
        scores = F.conv1d(attributions.view(1, 1, -1), kernel)  # [1, 1, L-block_size+1]
        scores = scores.view(-1)  # [L - block_size + 1]

        # greedy selection of non-overlapping blocks
        starts_sorted = torch.argsort(scores, descending=True)
        used = torch.zeros(L, dtype=torch.bool, device=attributions.device)
        blocks = []

        for s in starts_sorted.tolist():
            if len(blocks) >= num_blocks:
                break
            block_range = torch.arange(s, s + block_size, device=attributions.device)
            if used[block_range].any():
                continue
            blocks.append(block_range)
            used[block_range] = True

        if not blocks:
            return torch.tensor([], dtype=torch.long, device=attributions.device)
        return torch.cat(blocks)

    def _closest_embedding_indices(self, emb_weight: torch.Tensor,
                                   vecs: torch.Tensor) -> torch.Tensor:
        """
        For each row in `vecs` (K, D), find the index in `emb_weight` (V, D)
        with smallest squared Euclidean distance.
        Returns LongTensor [K].
        """
        # ||v - w||^2 = ||v||^2 + ||w||^2 - 2 v·w
        v2 = (vecs ** 2).sum(dim=1, keepdim=True)  # [K, 1]
        w2 = (emb_weight ** 2).sum(dim=1).unsqueeze(0)  # [1, V]
        dot = vecs @ emb_weight.t()  # [K, V]
        dist2 = v2 + w2 - 2 * dot  # [K, V]
        _, idx = dist2.min(dim=1)  # [K]
        return idx.long()



class GreedyBlockTrainer:
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 epochs: int = 5,
                 loss: torch.nn.Module = None,
                 scheduler: _LRScheduler = None,
                 greedy_block_augmentor: GreedyBlockAugmentor = None) -> None:
        """
            Create PyTorch trainer.

            Parameters
            ----------
            optimizer : torch.optim.Optimizer
                Optimizer to use for training the model.
            epochs : int, optional
                Number of epochs, by default 5.
            loss : torch.nn.Module, optional
                Loss to minimize, by default BCEWithLogitsLoss.
            scheduler : _LRScheduler, optional
                Scheduler for the optimizer, by default None.
            greedy_block_augmentor : GreedyBlockAugmentor, optional
                If provided, GreedyBlock augmentation will be applied
                to (at most) one example per batch during training.
        """
        self._epochs = epochs
        self._optimizer = optimizer
        self._loss = loss if loss is not None else torch.nn.BCEWithLogitsLoss()
        self._scheduler = scheduler
        self._greedyblock = greedy_block_augmentor

        self.training_losses = []
        self.training_accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []

    def train(self, model: torch.nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            patience: int = None) -> torch.nn.Module:
        """
        Train model with given loaders and early stopping.
        Parameters
        ----------
        model : torch.nn.Module
            Pytorch model to be trained.
        train_loader : DataLoader
            Train data loader.
        val_loader : DataLoader
            Validation data loader.
        patience : int
            Number of epochs to wait before early stopping.
        Returns
        -------
        torch.nn.Module
            Trained model.
        """
        # Check model has .threshold
        if not hasattr(model, 'threshold'):
            raise AttributeError("Model must have a 'threshold' attribute for binary classification.")

        best_loss = float("inf")
        best_model = None
        patience_counter = 0
        for epoch in range(self._epochs):
            model = self.fit(model, train_loader)
            val_loss = self.validate(model, val_loader)
            if val_loss <= best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model)
                best_model = best_model.to(next(model.parameters()).device)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience is not None:
                if patience_counter >= patience:
                    print(f"Early stopping triggered. The validation losss hasn't improved for {patience_counter} epochs")
                    break
            print(
                f"Epoch {epoch}: val_loss = {val_loss}, best_loss = {best_loss}, patience_counter = {patience_counter}")
        return best_model

    def fit(self,
            model: torch.nn.Module,
            dataloader: DataLoader) -> torch.nn.Module:
        """
        Train model for one epoch with given loader.
        """
        device = next(model.parameters()).device
        model = model.train()
        running_loss = 0.0
        train_total = 0
        train_correct = 0
        num_batches = 0

        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)

            # --- GreedyBlock augmentation (one example per batch) ---
            if self._greedyblock is not None:
                x = self._greedyblock.augment_batch(model, x, y)

            self._optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            outputs = outputs.view(-1)
            y_flat = y.view(-1)

            loss = self._loss(outputs, y_flat.float())
            loss.backward()
            self._optimizer.step()

            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            y_preds = (probs >= model.threshold).int()

            train_total += y_flat.size(0)
            train_correct += (y_preds == y_flat).sum().item()
            num_batches += 1

        self.training_losses.append(running_loss / num_batches)
        self.training_accuracies.append(train_correct / train_total)

        if self._scheduler is not None:
            self._scheduler.step()
        return model

    def validate(self,
                 model: torch.nn.Module,
                 dataloader: DataLoader) -> float:
        """
        Validate model with given loader.
        """
        running_loss = 0
        val_total = 0
        val_correct = 0
        device = next(model.parameters()).device
        model = model.eval()
        num_batches = 0
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                outputs = outputs.view(-1)
                y_flat = y.view(-1)
                loss = self._loss(outputs, y_flat.float())
                running_loss += loss.item()
                probs = torch.sigmoid(outputs)
                y_preds = (probs >= model.threshold).int()
                val_total += y_flat.size(0)
                val_correct += (y_preds == y_flat).sum().item()
                num_batches += 1

            val_loss = running_loss / num_batches
            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_correct / val_total)
        return val_loss
