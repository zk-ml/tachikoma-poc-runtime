import torch
import tvm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tvm import relay

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name, return_dict=False)

text = "I'm sorry, Dave. [MASK]"
inputs = tokenizer(text, return_tensors="pt")["input_ids"]

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

torch_output = model(inputs)[0].numpy()

traced_model = torch.jit.trace(model, inputs)
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)

shape_list = [
    (i.debugName().split(".")[0], i.type().sizes())
    for i in list(traced_model.graph.inputs())[1:]
]
print(traced_model)
print(shape_list)
mod, params = relay.frontend.pytorch.from_pytorch(
    traced_model, shape_list, default_dtype="int8"
)

with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=65536.0):
    mod = relay.quantize.quantize(mod, params)

print(mod)

target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target=target, params=params)

from tvm.contrib import graph_executor

input_name = "input_ids"

m = graph_executor.GraphModule(lib["default"](dev))
# Set inputs
m.set_input(input_name, tvm.nd.array(inputs.numpy()))
# Execute
m.run()
# Get outputs
tvm_output = m.get_output(0).numpy()

print((tvm_output - torch_output).abs())
