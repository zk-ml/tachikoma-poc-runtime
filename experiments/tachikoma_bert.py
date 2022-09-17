import numpy as np
import tvm
from tvm.relay.op.contrib import tachikoma
from tvm import relay
from tvm.relay import build_module
import tvm.relay.testing
import torch
import torch.onnx
import tvm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tvm import relay
import numpy as np

input_name = "input_ids"
model_name = "bert-base-uncased" # "kssteven/ibert-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(
    model_name, return_dict=False, torchscript=True # quant_mode=True, 
)

text = "I'm sorry, Dave. [MASK]"
inputs = tokenizer(text, return_tensors="pt")["input_ids"]

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

torch_output = model(inputs)[0].numpy()

traced_model = torch.jit.script(model, inputs)
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
    traced_model, shape_list, keep_quantized_weight=True
)

device = tvm.cpu()
target = "llvm"
dtype = "float32"
print(mod["main"].astext(show_meta_data=False), "\n")
mod = tachikoma.partition_for_tachikoma(mod, params)
print(mod["main"].astext(show_meta_data=False), "\n")
print(mod.get_global_vars())
print(type(mod))

with tvm.transform.PassContext(opt_level=1):
    lib, bldmod = build_module.build_with_bldmod(mod, target=target, params=params)

path_set = tvm.get_global_func("runtime.TachikomaSetExportPath")

explib = bldmod._get_module()
rmod = lib["default"](device)
#print(type(explib))
#print(type(lib["default"]))
print(lib)
print(explib)
print(type(explib))

rt_mod = tvm.contrib.graph_executor.GraphModule(rmod)

for i in range(2):
    path_set(explib, f"/data/tachikoma_results/serialized_{i}.ndarray")
    
    for name, data in lib.get_params().items():
        print(name, data.shape)
        data = tvm.nd.array(data.numpy() + i)
        rt_mod.set_input(name, data)
    rt_mod.run()

    out = rt_mod.get_output(0)

