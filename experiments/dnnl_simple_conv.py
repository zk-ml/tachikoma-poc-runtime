import numpy
import tvm
from tvm.relay.op.contrib.dnnl import pattern_table
from tvm import relay
import os

dshape = (64, 1, 32, 32)
kshape = (1, 1, 1, 1)
eltype = "float32"
scale = 100

data = numpy.random.uniform(-scale, scale, size=dshape).astype(eltype)
kern = numpy.random.uniform(-scale, scale, size=kshape).astype(eltype)

w = relay.var("weight", shape=kshape, dtype="float32")
x = relay.var("x", shape=dshape, dtype=eltype)
y = relay.nn.conv2d(
    x, w, padding=(1, 1), dilation=(1, 1), groups=1, channels=1, kernel_size=(1, 1)
)
z = relay.nn.relu(y)

mod = tvm.IRModule()
mod["main"] = relay.Function([x, w], z)

mod = relay.transform.MergeComposite(pattern_table())(mod)
mod = relay.transform.AnnotateTarget(["dnnl"])(mod)  # Output: Figure 2
mod = relay.transform.MergeCompilerRegions()(mod)  # Output: Figure 3
mod = relay.transform.PartitionGraph()(mod)  # Output: Figure 4

graph, module, params = relay.build(mod, target="llvm")

with open('graph.json', 'w') as f:
    f.write(graph)