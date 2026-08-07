import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.manifold import TSNE
import numpy as np
import argparse
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from dataset import AbusiveLangDataset
from model import MultiTaskContrastiveAbuseModel
from contrastive_loss import ContrastiveRationaleLoss
from metrics import ClassificationMetrics
from train import Trainer
from rationale_metrics import evaluate_rationale_metrics
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, average_precision_score
import os
import random
from sklearn.metrics import roc_curve, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from explain import display_rationale, batch_explain
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = {0: "0", 1: "1"}
TOKENIZER_DIR = 'tokenizer_dir'

def save_model_and_tokenizer(model, tokenizer, save_path, tokenizer_dir):
    torch.save(model.state_dict(), save_path)
    tokenizer.save_pretrained(tokenizer_dir)
    logger.info(f"Model saved to {save_path}, tokenizer saved to {tokenizer_dir}")

def load_model_and_tokenizer(model_class, model_name, save_path, tokenizer_dir, device):
    tokenizer_path = tokenizer_dir if os.path.isdir(tokenizer_dir) else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = model_class(model_name).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, tokenizer


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


    cm = confusion_matrix(y_true, y_pred)
    cm_decimal = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(3, 3))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_decimal, display_labels=[1, 0])
    disp.plot(cmap="Reds", colorbar=False, ax=plt.gca(),
              values_format='.2f')
    plt.xlabel("Predicted label", fontsize=12)
    plt.ylabel("True label", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix1.pdf", dpi=600, bbox_inches="tight")
    plt.show()

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(3, 3))
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/Roc_curve1.pdf", dpi=600, bbox_inches="tight")
    plt.show()

CLASS_NAMES = {0: "not-offensive", 1: "offensive"}
def plot_tsne(embeddings, labels, class_names):
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(3, 3))
    for i, name in class_names.items():
        idxs = (labels == i)
        plt.scatter(X_2d[idxs, 0], X_2d[idxs, 1], label=name, alpha=0.6)

    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/multi_t-SNE1.pdf", dpi=600, bbox_inches="tight")
    plt.show()



def plot_precision_recall(y_true, y_probs, class_names):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt

    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    plt.figure(figsize=(3, 3))
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/Precision-Recall1.pdf", dpi=600, bbox_inches="tight")
    plt.show()



def save_probs_and_labels_model2(probs, labels, save_path="outputs/contrastive_probs_and_labels.npz"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    probs = np.array(probs)
    labels = np.array(labels)
    np.savez_compressed(save_path, probs=probs, labels=labels)

    logger.info(f"[Model 2] Saved probabilities and labels to: {save_path}")


def main():
    set_seed(42) # We set the seed to 42 for reproducibility
    parser = argparse.ArgumentParser(description="Contrastive Multi-task Abusive Language Detection (Span Rationale)")
    subparsers = parser.add_subparsers(dest="command")


    parser_train = subparsers.add_parser("train", help="Train and validate the model")
    parser_train.add_argument("--train_csv", default="train.csv")
    parser_train.add_argument("--dev_csv", default="dev.csv")
    parser_train.add_argument("--save_path", default="best_model.pth")
    parser_train.add_argument("--max_len", type=int, default=MAX_LEN)
    parser_train.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser_train.add_argument("--epochs", type=int, default=EPOCHS)
    parser_train.add_argument("--lr", type=float, default=3e-5) 


    parser_eval = subparsers.add_parser("eval", help="Evaluate the model and plot visualizations")
    parser_eval.add_argument("--test_csv", default="test.csv")
    parser_eval.add_argument("--save_path", default="best_model.pth")
    parser_eval.add_argument("--max_len", type=int, default=MAX_LEN)
    parser_eval.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser_eval.add_argument("--plot", action="store_true")


    parser_explain = subparsers.add_parser("explain", help="Display model rationale for a given input text")
    parser_explain.add_argument("--input_text", required=True, help="Text to explain")
    parser_explain.add_argument("--save_path", default="best_model.pth")
    parser_explain.add_argument("--max_len", type=int, default=MAX_LEN)


    parser_batch = subparsers.add_parser("explain-batch", help="Batch rationale explanations (HTML output)")
    parser_batch.add_argument("--infile", required=True, help="CSV input file")
    parser_batch.add_argument("--outfile", required=True, help="HTML output file")
    parser_batch.add_argument("--text_column", default="text")
    parser_batch.add_argument("--save_path", default="best_model.pth")
    parser_batch.add_argument("--max_len", type=int, default=MAX_LEN)

    args = parser.parse_args()

    if args.command == "train":
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        train_dataset = AbusiveLangDataset(args.train_csv, tokenizer, args.max_len)
        dev_dataset = AbusiveLangDataset(args.dev_csv, tokenizer, args.max_len)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
        model = MultiTaskContrastiveAbuseModel(MODEL_NAME).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        ce_loss = nn.CrossEntropyLoss()
        contrastive_loss = ContrastiveRationaleLoss()
        metrics = ClassificationMetrics(CLASS_NAMES)
        trainer = Trainer(
            model, optimizer, ce_loss, DEVICE, contrastive_loss, metrics, CLASS_NAMES
        )
        best_f1 = 0
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            train_loss = trainer.train_somoff(train_loader)
            train_loss_eval, train_acc, train_prec, train_rec, train_f1, train_report, train_auroc = trainer.eval_somoff(train_loader)
            print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f} | Train F1: {train_f1:.4f}")
            dev_loss, dev_acc, dev_prec, dev_rec, dev_f1, dev_report, dev_auroc = trainer.eval_somoff(dev_loader)
            print(f"Dev loss: {dev_loss:.4f}")
            print(f"Dev accuracy: {dev_acc:.4f} | Precision: {dev_prec:.4f} | Recall: {dev_rec:.4f} | F1: {dev_f1:.4f}")
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                save_model_and_tokenizer(model, tokenizer, args.save_path, TOKENIZER_DIR)


    elif args.command == "eval":
        if not os.path.exists(TOKENIZER_DIR):
            raise FileNotFoundError(
                f"Tokenizer directory '{TOKENIZER_DIR}' not found! Run training first or specify the correct path.")
        model, tokenizer = load_model_and_tokenizer(
            MultiTaskContrastiveAbuseModel, MODEL_NAME, args.save_path, TOKENIZER_DIR, DEVICE
        )
        test_dataset = AbusiveLangDataset(args.test_csv, tokenizer, args.max_len)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        ce_loss = nn.CrossEntropyLoss()
        contrastive_loss = ContrastiveRationaleLoss()
        metrics = ClassificationMetrics(CLASS_NAMES)
        trainer = Trainer(
            model, None, ce_loss, DEVICE, contrastive_loss, metrics, CLASS_NAMES
        )
        logger.info("\n--- Test Set Evaluation ---")
        test_results = trainer.eval_somoff(test_loader, return_probs=True, return_embs=True)
        test_loss, test_acc, test_prec, test_rec, test_f1, test_report, test_auroc, test_probs, test_labels, test_embs, test_labels_emb = test_results
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(f"Test accuracy: {test_acc:.4f}")
        logger.info(f"Test precision: {test_prec:.4f}")
        logger.info(f"Test recall: {test_rec:.4f}")
        logger.info(f"Test f1-score: {test_f1:.4f}")
        logger.info(f"Test AUROC: {test_auroc:.4f}")
        logger.info("Detailed classification report: %s \n", test_report)

        logger.info("\n-- Rationale plausibility and faithfulness metrics --")
        for split_file in ["train.csv", "dev.csv", "test.csv"]:
            logger.info(f"\n[Metrics for {split_file}]")
            evaluate_rationale_metrics(
                model,
                tokenizer,
                split_file,
                DEVICE,
                max_len=args.max_len,
                batch_size=args.batch_size,
                verbose=True,
                calc_faithfulness=True
            )

        plot_confusion_matrix(test_labels, np.array(test_probs) > 0.5, CLASS_NAMES)
        plot_precision_recall(test_labels, test_probs, CLASS_NAMES)
        plot_tsne(np.array(test_embs), np.array(test_labels_emb), CLASS_NAMES)
        plot_roc_curve(test_labels, test_probs)
        save_probs_and_labels_model2(test_probs, test_labels)
    elif args.command == "explain":
        if not os.path.exists(TOKENIZER_DIR):
            raise FileNotFoundError(f"Tokenizer directory '{TOKENIZER_DIR}' not found! Run training first or specify the correct path.")
        model, tokenizer = load_model_and_tokenizer(
            MultiTaskContrastiveAbuseModel, MODEL_NAME, args.save_path, TOKENIZER_DIR, DEVICE
        )
        display_rationale(model, tokenizer, args.input_text, DEVICE, args.max_len)

    elif args.command == "explain-batch":
        if not os.path.exists(TOKENIZER_DIR):
            raise FileNotFoundError(
                f"Tokenizer directory '{TOKENIZER_DIR}' not found! Run training first or specify the correct path.")
        model, tokenizer = load_model_and_tokenizer(
            MultiTaskContrastiveAbuseModel, MODEL_NAME, args.save_path, TOKENIZER_DIR, DEVICE
        )
        batch_explain(model, tokenizer, args.infile, args.outfile, DEVICE, args.max_len, text_column=args.text_column)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()






