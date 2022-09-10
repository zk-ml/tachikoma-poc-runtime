import torch
import torch.onnx
import tvm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tvm import relay
import numpy as np
import onnx

pytorch = False

input_name = "input_ids"
model_name = "kssteven/ibert-roberta-base"  # "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(
    model_name, return_dict=False, quant_mode=True, torchscript=True
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

if pytorch:
    mod, params = relay.frontend.pytorch.from_pytorch(
        traced_model, shape_list, keep_quantized_weight=True
    )

else:
    model_path = "ibert.onnx"
    torch.onnx.export(
        traced_model,  # model being run
        inputs,  # model input (or a tuple for multiple inputs)
        model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=[input_name],  # the model's input names
        output_names=["output"],  # the model's output names
    )

    onnx_model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_list)

# with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=127.0):
#    mod = relay.quantize.quantize(mod, params)

print(mod)

target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target=target, params=params)

from tvm.contrib import graph_executor

m = graph_executor.GraphModule(lib["default"](dev))
# Set inputs
m.set_input(input_name, tvm.nd.array(inputs.numpy()))
# Execute
m.run()
# Get outputs
tvm_output = m.get_output(0).numpy()

print(np.absolute(tvm_output - torch_output))
