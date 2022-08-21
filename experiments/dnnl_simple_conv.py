import numpy as np
import tvm
from tvm.relay.op.contrib.dnnl import pattern_table
from tvm import relay
from tvm.relay import testing
import tvm.contrib.graph_executor as runtime

dtype = "float32"
scale = 1
"""
dshape = (64, 1, 32, 32)
kshape = (1, 1, 1, 1)
eltype = "float32"
scale = 100

data = numpy.random.uniform(-scale, scale, size=dshape).astype(eltype)
kern = numpy.random.uniform(-scale, scale, size=kshape).astype(eltype)

w = relay.var("weight", shape=kshape, dtype=eltype)
x = relay.var("x", shape=dshape, dtype=eltype)
y = relay.nn.conv2d(
    x, w, padding=(1, 1), dilation=(1, 1), groups=1, channels=1, kernel_size=(1, 1)
)
z = relay.nn.relu(y)

mod = tvm.IRModule()
mod["main"] = relay.Function([x, w], z)
"""

da = np.random.uniform(-scale, scale, size=(1, 10)).astype(dtype)
db = np.random.uniform(-scale, scale, size=(1, 10)).astype(dtype)

a = relay.var("a", shape=(1, 10), dtype=dtype)
b = relay.var("b", shape=(1, 10), dtype=dtype)
out = relay.add(a, b)

func = relay.Function([a, b], out)
mod = tvm.IRModule.from_expr(func)

patterns = pattern_table()
# print(patterns)

mod = relay.transform.MergeComposite(patterns)(mod)
print(mod["main"].astext(show_meta_data=False), "\n")
mod = relay.transform.AnnotateTarget(["dnnl"])(mod)
print(mod["main"].astext(show_meta_data=False), "\n")
mod = relay.transform.MergeCompilerRegions()(mod)
print(mod["main"].astext(show_meta_data=False), "\n")
mod = relay.transform.PartitionGraph()(mod)
print(mod["main"].astext(show_meta_data=False), "\n")

with tvm.transform.PassContext(opt_level=0):
    graph, module, params = relay.build(mod, target="llvm", params={"a": da, "b": db})

with open("graph.json", "w") as f:
    f.write(graph)


print(graph, module, params)
