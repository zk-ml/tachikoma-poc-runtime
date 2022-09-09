import torch
import tvm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tvm import relay

tokenizer = AutoTokenizer.from_pretrained("kssteven/ibert-roberta-base")
model = AutoModelForMaskedLM.from_pretrained(
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

print(shape_list)
mod, params = relay.frontend.pytorch.from_pytorch(
    traced_model, shape_list, default_dtype="float32"
)

target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target=target, params=params)


from tvm.contrib import graph_executor

input_name = "input_ids"

dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# Set inputs
m.set_input(input_name, tvm.nd.array(inputs.numpy().astype(dtype)))
# Execute
m.run()
# Get outputs
tvm_output = m.get_output(0)

print(tvm_output)
