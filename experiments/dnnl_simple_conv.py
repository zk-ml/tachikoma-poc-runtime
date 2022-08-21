import numpy
import tvm
from tvm.relay.op.contrib.dnnl import pattern_table
from tvm import relay

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

a = relay.var("a", shape=(1, 10), dtype="float32")
b = relay.var("b", shape=(1, 10), dtype="float32")
c = relay.var("c", shape=(1, 10), dtype="float32")
out = relay.add(a, b)
out = relay.add(out, c)

func = relay.Function([a, b, c], out)
mod = tvm.IRModule.from_expr(func)

patterns = pattern_table()
# print(patterns)

mod = relay.transform.MergeComposite(patterns)(mod)
mod = relay.transform.AnnotateTarget(["dnnl"])(mod)
mod = relay.transform.MergeCompilerRegions()(mod)
mod = relay.transform.PartitionGraph()(mod)

print(mod)

with tvm.transform.PassContext(opt_level=0):
    graph, module, params = relay.build(mod, target="llvm")

with open("graph.json", "w") as f:
    f.write(graph)
