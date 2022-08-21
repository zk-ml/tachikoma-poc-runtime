import numpy
import tvm
from tvm.relay.op.contrib.dnnl import pattern_table
from tvm import relay
import os

target = "llvm"

batch_size = 1
dtype = "float32"
model_name = "resnet-18"
log_file = "%s.log" % model_name
graph_opt_sch_file = "%s_graph_opt.log" % model_name

# Set the input name of the graph
# For ONNX models, it is typically "0".
input_name = "data"

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape

dshape = (64, 1, 32, 32)
kshape = (1, 1, 1, 1)
eltype = "uint16"
scale = 1

data = numpy.random.uniform(-scale, scale, size=dshape).astype(eltype)
kern = numpy.random.uniform(-scale, scale, size=kshape).astype(eltype)

w = tvm.relay.const(kern)
x = tvm.relay.var("x", shape=dshape, dtype=eltype)
y = tvm.relay.nn.conv2d(
    x, w, padding=(1, 1), dilation=(1, 1), groups=1, channels=1, kernel_size=(1, 1)
)

mod = tvm.IRModule()
mod["main"] = relay.Function([x], y)
#mod, params, data_shape, out_shape = get_network(model_name, batch_size)

mod = relay.transform.MergeComposite(pattern_table())(mod)
mod = relay.transform.AnnotateTarget(["dnnl"])(mod)  # Output: Figure 2
mod = relay.transform.MergeCompilerRegions()(mod)  # Output: Figure 3
mod = relay.transform.PartitionGraph()(mod)  # Output: Figure 4

graph, module, params = relay.build(mod, target="llvm")

import json
with open('graph.json', 'w') as f:
    f.write(graph)