import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from dataset import AbusiveLangDataset
from torch.utils.data import DataLoader

def token_mask_from_span(start_idx, end_idx, seq_len):
    mask = np.zeros(seq_len, dtype=int)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    mask[start_idx:end_idx+1] = 1
    return mask

def iou_f1_score(pred_mask, gold_mask):
    intersection = (pred_mask & gold_mask).sum()
    union = (pred_mask | gold_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    iou = intersection / union
    return iou
def token_f1(pred_mask, gold_mask):
    tp = (pred_mask & gold_mask).sum()
    fp = (pred_mask & ~gold_mask).sum()
    fn = (~pred_mask & gold_mask).sum()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1
def auprc_score(soft_scores, gold_mask):
    precision, recall, _ = precision_recall_curve(gold_mask, soft_scores)
    return auc(recall, precision)

def comprehensiveness_and_sufficiency(model, tokenizer, text, rationale_span, device, pred_class_idx, max_len=128):
    inputs = tokenizer(
        text, return_tensors='pt', truncation=True, max_length=max_len, padding='max_length'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits, *_ = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.softmax(logits, dim=1).squeeze()
        orig_prob = probs[pred_class_idx].item()
    start, end = rationale_span
    text_minus = text[:start] + text[end:]
    inputs_minus = tokenizer(
        text_minus, return_tensors='pt', truncation=True, max_length=max_len, padding='max_length'
    )
    inputs_minus = {k: v.to(device) for k, v in inputs_minus.items()}
    with torch.no_grad():
        logits_minus, *_ = model(inputs_minus['input_ids'], inputs_minus['attention_mask'])
        probs_minus = torch.softmax(logits_minus, dim=1).squeeze()
        minus_prob = probs_minus[pred_class_idx].item()
    comprehensiveness = orig_prob - minus_prob
    text_rationale = text[start:end]
    if text_rationale.strip() == "":
        return None, None
    inputs_only = tokenizer(
        text_rationale, return_tensors='pt', truncation=True, max_length=max_len, padding='max_length'
    )
    inputs_only = {k: v.to(device) for k, v in inputs_only.items()}
    with torch.no_grad():
        logits_only, *_ = model(inputs_only['input_ids'], inputs_only['attention_mask'])
        probs_only = torch.softmax(logits_only, dim=1).squeeze()
        only_prob = probs_only[pred_class_idx].item()
    sufficiency = orig_prob - only_prob

    return comprehensiveness, sufficiency

def evaluate_rationale_metrics(
    model,
    tokenizer,
    data_path,
    device,
    max_len=128,
    batch_size=16,
    verbose=True,
    calc_faithfulness=True,
):
    

    model.eval()
    dataset = AbusiveLangDataset(data_path, tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=batch_size)
    ious, iou_f1s, precisions, recalls, f1s, auprcs = [], [], [], [], [], []
    comprehensivenesses, sufficiencies = [], []
    texts = dataset.df['text'].tolist()

    #has_offsets = hasattr(dataset, "return_offsets") and dataset.return_offsets

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gold_start = batch['rationale_start'].cpu().numpy()
            gold_end = batch['rationale_end'].cpu().numpy()
            labels = batch['label'].cpu().numpy()

            offset_mappings = batch.get('offset_mapping', None) if hasattr(batch, 'get') else None

            if 'index' in batch:
                batch_indices = batch['index'].cpu().numpy()
            else:
                start_idx = batch_idx * batch_size
                batch_indices = np.arange(start_idx, min(start_idx + input_ids.shape[0], len(texts)))
            batch_texts = [texts[i] for i in batch_indices]

            logits, start_logits, end_logits, *_ = model(input_ids, attention_mask)
            pred_start = torch.argmax(start_logits, dim=1).cpu().numpy()
            pred_end = torch.argmax(end_logits, dim=1).cpu().numpy()
            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for j in range(input_ids.shape[0]):
                seq_len = input_ids[j].shape[0]
                pred_mask = token_mask_from_span(pred_start[j], pred_end[j], seq_len)
                gold_mask = token_mask_from_span(gold_start[j], gold_end[j], seq_len)
                iou = iou_f1_score(pred_mask, gold_mask)
                ious.append(iou)
                iou_f1s.append(1.0 if iou >= 0.5 else 0.0)
                p, r, f = token_f1(pred_mask, gold_mask)
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)
                soft_scores = np.maximum(start_probs[j], end_probs[j])
                auprcs.append(auprc_score(soft_scores, gold_mask))


                if calc_faithfulness:
                    text = batch_texts[j]

                    try:

                        if offset_mappings is not None:
                            offsets = offset_mappings[j]
                            char_start = offsets[pred_start[j]][0]
                            char_end = offsets[pred_end[j]][1]
                        else:

                            char_start, char_end = 0, len(text)
                        if char_end > char_start:
                            #pred_class_idx = int(labels[j])
                            pred_class_idx = int(torch.argmax(logits[j]).item())
                            comp, suff = comprehensiveness_and_sufficiency(
                                model, tokenizer, text, (char_start, char_end), device, pred_class_idx, max_len
                            )
                            if comp is not None: comprehensivenesses.append(comp)
                            if suff is not None: sufficiencies.append(suff)
                    except Exception as e:
                        print(f"Faithfulness error: {e} on sample {batch_indices[j] if 'index' in batch else j}")
                        continue

    results = {
        "IOU F1": np.mean(iou_f1s),
        "Token Precision": np.mean(precisions),
        "Token Recall": np.mean(recalls),
        "Token F1": np.mean(f1s),
        "AUPRC": np.mean(auprcs),
        "Comprehensiveness": np.mean(comprehensivenesses) if comprehensivenesses else None,
        "Sufficiency": np.mean(sufficiencies) if sufficiencies else None,
    }
    if verbose:
        print("\nRationale Metrics:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")
    return results
