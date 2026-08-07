import torch
import torch
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from visualization_utils import plot_token_heatmap
def _get_valid_indices(attention_mask, tokenizer):
    valid_mask = attention_mask[0].cpu().numpy().astype(bool)

    valid_mask[0] = 0
    if tokenizer.sep_token_id is not None and len(valid_mask) > 1:
        valid_mask[-1] = 0
    return valid_mask

def display_rationale(model, tokenizer, text, device, max_len=128, class_names=None):
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_len,
        padding='max_length',
        return_offsets_mapping=True
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offsets = encoding['offset_mapping'][0].tolist()

    with torch.no_grad():
        logits, start_logits, end_logits, _, _ = model(input_ids, attention_mask)

        valid_mask = _get_valid_indices(attention_mask, tokenizer)
        start_logits_np = start_logits[0].cpu().numpy()
        end_logits_np = end_logits[0].cpu().numpy()
        start_logits_np[~valid_mask] = -1e9
        end_logits_np[~valid_mask] = -1e9
        start_idx = start_logits_np.argmax()
        end_idx = end_logits_np.argmax()

        cls_pred = torch.argmax(logits, dim=1).item()
        pred_label = class_names[cls_pred] if class_names else str(cls_pred)


    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    start_char, _ = offsets[start_idx]
    _, end_char = offsets[end_idx]
    if start_char >= end_char:
        result = text
    else:
        result = text[:start_char] + f"\033[91m{text[start_char:end_char]}\033[0m" + text[end_char:]
    logger.info(f"Input Text:\n{text}\n")
    logger.info(f"Predicted Label: {pred_label}\n")
    logger.info("Rationale Highlighted (span-level):\n")
    logger.info(result)
    logger.info("\nLegend: red = rationale span\n")
    logger.info(f"\nThe text is offensive because of: \033[91m{text[start_char:end_char]}\033[0m\n")

def rationale_to_html(model, tokenizer, text, device, max_len=128):
    model.eval()
    encoding = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=max_len,
        padding='max_length',
        return_offsets_mapping=True
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offsets = encoding['offset_mapping'][0].tolist()
    with torch.no_grad():
        logits, start_logits, end_logits, _, _ = model(input_ids, attention_mask)
        valid_mask = _get_valid_indices(attention_mask, tokenizer)
        start_logits_np = start_logits[0].cpu().numpy()
        end_logits_np = end_logits[0].cpu().numpy()
        start_logits_np[~valid_mask] = -1e9
        end_logits_np[~valid_mask] = -1e9
        start_idx = start_logits_np.argmax()
        end_idx = end_logits_np.argmax()
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    start_char, _ = offsets[start_idx]
    _, end_char = offsets[end_idx]
    if start_char >= end_char:
        result = text
    else:
        result = (
            text[:start_char] +
            f'<span style="background-color: yellow">{text[start_char:end_char]}</span>' +
            text[end_char:]
        )
    return result

def batch_explain(model, tokenizer, infile, outfile, device, max_len=128, text_column="text"):
    import pandas as pd
    df = pd.read_csv(infile)
    htmls = []
    for text in df[text_column]:
        html = rationale_to_html(model, tokenizer, text, device, max_len)
        htmls.append(html)
    df["rationale_html"] = htmls
    df.to_html(outfile, escape=False, index=False)
    logger.info(f"Batch explanations with highlighted rationales saved to {outfile}")


