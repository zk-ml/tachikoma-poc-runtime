import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import numpy as np
import tvm
from tvm.relay.op.contrib import tachikoma
from tvm import relay
from tvm.relay import build_module
import tvm.relay.testing

QUANT = True
DEBUG = False

class LeNet_Small_Quant(nn.Module):
    def __init__(self):
        super(LeNet_Small_Quant, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=5, stride=1, bias=False
        )
        self.act1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=False
        )
        self.act2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=4, stride=1, bias=False
        )  # LeNet use 5 for 32x32. For 28x28, we adjust to 4.
        self.act3 = nn.ReLU()
        self.linear1 = nn.Linear(in_features=480, out_features=84)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=84, out_features=10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        if QUANT:
            x = self.quant(x)
        x = self.conv1(x)
        x = self.act1(x)
        
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        #x = self.conv3(x)
        #x = self.act3(x)
        #x = x.reshape(x.size(0), -1)
        #x = self.linear1(x)
        #x = self.act4(x)
        #x = self.linear2(x)
        if QUANT:
            x = self.dequant(x)
        return x

    def dump_feat_param(self):
        dummy_image = torch.tensor(np.ones((1, 3, 32, 32))).float()
        x = self.quant(dummy_image)
        conv1 = self.conv1(x)
        act1 = self.act1(conv1)
        pool1 = self.pool1(act1)
        conv2 = self.conv2(pool1)
        act2 = self.act2(conv2)
        pool2 = self.pool2(act2)
        conv3 = self.conv3(pool2)
        act3 = self.act3(conv3)
        view_output = act3.reshape(act3.size(0), -1)
        linear1 = self.linear1(view_output)
        act4 = self.act4(linear1)
        linear2 = self.linear2(act4)
        output = self.dequant(linear2)
        feature_quantize_parameters = {
            "x_q_scale": x.q_scale(),
            "x_q_zero_point": x.q_zero_point(),
            "conv1_q_scale": conv1.q_scale(),
            "conv1_q_zero_point": conv1.q_zero_point(),
            "act1_q_scale": act1.q_scale(),
            "act1_q_zero_point": act1.q_zero_point(),
            "pool1_q_scale": pool1.q_scale(),
            "pool1_q_zero_point": pool1.q_zero_point(),
            "conv2_q_scale": conv2.q_scale(),
            "conv2_q_zero_point": conv2.q_zero_point(),
            "act2_q_scale": act2.q_scale(),
            "act2_q_zero_point": act2.q_zero_point(),
            "pool2_q_scale": pool2.q_scale(),
            "pool2_q_zero_point": pool2.q_zero_point(),
            "conv3_q_scale": conv3.q_scale(),
            "conv3_q_zero_point": conv3.q_zero_point(),
            "act3_q_scale": act3.q_scale(),
            "act3_q_zero_point": act3.q_zero_point(),
            "linear1_q_scale": linear1.q_scale(),
            "linear1_q_zero_point": linear1.q_zero_point(),
            "act4_q_scale": act4.q_scale(),
            "act4_q_zero_point": act4.q_zero_point(),
            "linear2_q_scale": linear2.q_scale(),
            "linear2_q_zero_point": linear2.q_zero_point(),
        }
        return feature_quantize_parameters

    def quant_input(self, x):
        x = torch.tensor(x).float()
        x_quant = self.quant(x)
        return x_quant.int_repr().numpy(), x_quant.q_scale(), x_quant.q_zero_point()


model = LeNet_Small_Quant()
ishape = (1, 3, 32, 32)
pt_inp = torch.tensor(np.random.uniform(-1, 1, size=ishape)).float()

if QUANT:
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    print(model.qconfig)
    torch.quantization.prepare(model, inplace=True)
    for _ in range(100):
        model(torch.tensor(np.random.uniform(-1, 1, size=ishape)).float())
    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

script_module = torch.jit.trace(model, pt_inp).eval()

with torch.no_grad():
    pt_result = script_module(pt_inp).numpy()

print(script_module)

input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shapes = [(input_name, ishape)]
_mod, params = relay.frontend.from_pytorch(
    script_module, input_shapes, keep_quantized_weight=True
)

print(type(params))

device = tvm.cpu()
target = "llvm"
print(_mod["main"].astext(show_meta_data=False), "\n")
mod = tachikoma.partition_for_tachikoma(_mod, params)
print(mod["main"].astext(show_meta_data=False), "\n")
print(mod.get_global_vars())
print(type(mod))

if DEBUG:
    from tvm.contrib.debugger.debug_executor import GraphModuleDebug
    with tvm.transform.PassContext(opt_level=1):
        lib = relay.build(mod, target=target, params=params)
    lib.export_library("./debug.so")
    m = GraphModuleDebug(
        lib["debug_create"]("default", device),
        [device],
        lib.graph_json,
        dump_root="/tmp/tvmdbg",
    )
    # set inputs
    m.set_input(input_name, tvm.nd.array(np.random.uniform(-1, 1, ishape).astype("float32")))
    m.set_input(**params)
    # execute
    m.run()
    tvm_out = m.get_output(0).numpy()
else:

    with tvm.transform.PassContext(opt_level=1):
        func = relay.create_executor("graph", mod=mod, device=device, target=target).evaluate()

    with tvm.transform.PassContext(opt_level=1):
        func_ref = relay.create_executor("graph", mod=_mod, device=device, target=target).evaluate()

    print(params.keys())

    for i in range(3):
        input_dict = {input_name: np.random.uniform(-1, 1, ishape).astype("float32")}
        pred = func(**input_dict, **params)
        actual = func_ref(**input_dict, **params)
        err = (pred.numpy() - actual.numpy()).mean()
        print(f"iter {i}: err {err}")
