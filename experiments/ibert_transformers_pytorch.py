import torch
import tvm
from transformers import AutoTokenizer, IBertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("kssteven/ibert-roberta-base")
model = IBertForSequenceClassification.from_pretrained(
    "kssteven/ibert-roberta-base", return_dict=False
)

text = "I'm sorry, Dave."
inputs = tokenizer(text, return_tensors="pt")["input_ids"]

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

res = model(inputs)
print(res)

traced_model = torch.jit.trace(model, inputs)
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)

shape_list = [
    (i.debugName().split(".")[0], i.type().sizes())
    for i in list(traced_model.graph.inputs())[1:]
]

mod_bert, params_bert = tvm.relay.frontend.pytorch.from_pytorch(
    traced_model, shape_list, default_dtype="float32"
)
