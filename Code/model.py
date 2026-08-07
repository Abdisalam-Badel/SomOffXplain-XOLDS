import torch
import torch.nn as nn
from transformers import AutoModel
import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskContrastiveAbuseModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.rationale_start_head = nn.Linear(hidden_size, 1)
        self.rationale_end_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, rationale_start=None, rationale_end=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        logits = self.classifier(cls_output)
        rationale_start_logits = self.rationale_start_head(sequence_output).squeeze(-1)
        rationale_end_logits = self.rationale_end_head(sequence_output).squeeze(-1)

        span_reprs = []
        if rationale_start is not None and rationale_end is not None:
            for b in range(sequence_output.size(0)):
                s_idx = rationale_start[b]
                e_idx = rationale_end[b]
                if s_idx > e_idx:
                    s_idx, e_idx = e_idx, s_idx
                if s_idx == 0 and e_idx == 0:
                    span_emb = sequence_output[b, 0, :]
                else:
                    span_emb = sequence_output[b, s_idx:e_idx + 1, :].mean(dim=0)
                span_reprs.append(span_emb)
            span_reprs = torch.stack(span_reprs, dim=0)
        else:
            span_reprs = None

        return logits, rationale_start_logits, rationale_end_logits, cls_output, span_reprs

