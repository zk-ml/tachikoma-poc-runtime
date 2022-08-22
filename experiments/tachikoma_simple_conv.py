import numpy as np
import tvm
from tvm.relay.op.contrib.tachikoma import pattern_table
from tvm import relay
from tvm.relay import testing
import tvm.contrib.graph_executor as runtime
import os
import sys
import numpy as np
import tvm
from tvm.relay.backend import te_compiler
from tvm.relay.backend.runtime import Runtime
import tvm.relay.testing
from tvm import relay
from tvm import runtime as tvm_runtime
from tvm.relay import transform
from tvm.contrib import utils
import onnx
from tvm.contrib.download import download_testdata

dtype = "float32"
scale = 1
dshape = (64, 1, 32, 32)
kshape = (1, 1, 1, 1)

eltype = "float32"
scale = 100

data = np.random.uniform(-scale, scale, size=dshape).astype(eltype)
kern = np.random.uniform(-scale, scale, size=kshape).astype(eltype)

w = relay.var("weight", shape=kshape, dtype=eltype)
x = relay.var("x", shape=dshape, dtype=eltype)
y = relay.nn.conv2d(
    x, w, padding=(1, 1), dilation=(1, 1), groups=1, channels=1, kernel_size=(1, 1)
)
z = relay.nn.relu(y)

mod = tvm.IRModule()
mod["main"] = relay.Function([x, w], z)

patterns = pattern_table()
# print(patterns)

mod = relay.transform.MergeComposite(patterns)(mod)
print(mod["main"].astext(show_meta_data=False), "\n")
mod = relay.transform.AnnotateTarget(["tachikoma"])(mod)
print(mod["main"].astext(show_meta_data=False), "\n")
mod = relay.transform.MergeCompilerRegions()(mod)
print(mod["main"].astext(show_meta_data=False), "\n")
mod = relay.transform.PartitionGraph()(mod)
print(mod["main"].astext(show_meta_data=False), "\n")

map_inputs = {"weight": kern, "x": data}
with tvm.transform.PassContext(opt_level=0):
    graph, lib, params = relay.build(mod, target="llvm", params=map_inputs)

with open("graph.json", "w") as f:
    f.write(graph)

print(lib, params)

device = tvm.cpu()
rt_mod = tvm.contrib.graph_executor.create(graph, lib, device)
for name, data in params.items():
    rt_mod.set_input(name, data)
rt_mod.run()

out_shapes = [[64, 1, 32, 32]]

for idx, shape in enumerate(out_shapes):
    out = tvm.nd.empty(shape, device=device)
    out = rt_mod.get_output(idx, out)
    print(out)
