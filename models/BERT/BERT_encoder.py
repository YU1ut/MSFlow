import os

import torch.nn as nn


def load_bert(model_path):
    bert = BERT(model_path)
    bert.eval()
    bert.text_model.training = False
    for p in bert.parameters():
        p.requires_grad = False
    return bert


class BERT(nn.Module):
    def __init__(self, modelpath: str):
        super().__init__()

        from transformers import AutoModel, AutoTokenizer, logging

        logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        self.text_model = AutoModel.from_pretrained(modelpath)

    def forward(self, texts):
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(
            **encoded_inputs.to(self.text_model.device)
        ).last_hidden_state
        mask = encoded_inputs.attention_mask.to(dtype=bool)
        return output, mask
