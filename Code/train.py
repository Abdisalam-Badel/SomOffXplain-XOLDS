import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

from contrastive_loss import ContrastiveRationaleLoss
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, optimizer, ce_loss, device, contrastive_loss, metrics, class_names, alpha=0.05):
        self.model = model
        self.optimizer = optimizer
        self.ce_loss = ce_loss
        self.device = device
        self.contrastive_loss = contrastive_loss
        self.metrics = metrics
        self.class_names = class_names
        self.span_loss = nn.CrossEntropyLoss()
        self.alpha = alpha

    def train_somoff(self, dataloader):
        self.model.train()
        running_loss = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Train")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            rationale_start = batch['rationale_start'].to(self.device)
            rationale_end = batch['rationale_end'].to(self.device)

            logits, start_logits, end_logits, _, span_reprs = self.model(
                input_ids, attention_mask, rationale_start, rationale_end
            )
            loss_cls = self.ce_loss(logits, labels)
            loss_start = self.span_loss(start_logits, rationale_start)
            loss_end = self.span_loss(end_logits, rationale_end)

            loss_span_contrast = self.contrastive_loss.class_contrastive(span_reprs, labels)

            loss = loss_cls + (loss_start + loss_end) / 2 + self.alpha * loss_span_contrast

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            # We display the loss components every 20 batches for debugging
            if batch_idx % 20 == 0:
                logger.info(f"loss_cls: {loss_cls.item():.4f}, loss_start: {loss_start.item():.4f}, "
                      f"loss_end: {loss_end.item():.4f}, loss_span_contrast: {loss_span_contrast.item():.4f}")

        return running_loss / len(dataloader)

    def eval_somoff(self, dataloader, return_probs=False, return_embs=False):
        self.model.eval()
        running_loss = 0
        all_labels, all_preds, all_probs = [], [], []
        all_embs = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Eval"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                rationale_start = batch['rationale_start'].to(self.device)
                rationale_end = batch['rationale_end'].to(self.device)
                logits, start_logits, end_logits, cls_output, _ = self.model(
                    input_ids, attention_mask, rationale_start, rationale_end
                )

                loss_cls = self.ce_loss(logits, labels)
                loss_start = self.span_loss(start_logits, rationale_start)
                loss_end = self.span_loss(end_logits, rationale_end)
                loss = loss_cls + (loss_start + loss_end) / 2

                running_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                all_probs.extend(list(probs))
                all_preds.extend(list(preds))
                all_labels.extend(list(labels.cpu().numpy()))
                if return_embs:
                    all_embs.append(cls_output.cpu().numpy())
        out = [running_loss / len(dataloader)]
        acc, prec, rec, f1, report, auroc = self.metrics(all_labels, all_preds, all_probs)
        out.extend([acc, prec, rec, f1, report, auroc])
        if return_probs:
            out.append(np.array(all_probs))
            out.append(np.array(all_labels))
        if return_embs:
            out.append(np.concatenate(all_embs, axis=0))
            out.append(np.array(all_labels))
        return tuple(out)
