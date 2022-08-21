import numpy
import tvm
import tvm.relay
from tvm.relay.dataflow_pattern import is_op, wildcard
from tvm.relay.op.contrib.register import register_pattern_table
import tvm
from tvm import relay, autotvm
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_executor as runtime
import os 

def make_pattern(with_bias=True):
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op("nn.conv2d")(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    return is_op("nn.relu")(conv_out)


@register_pattern_table("dnnl")
def pattern_table():
    conv2d_bias_relu_pat = ("dnnl.conv2d_bias_relu", make_pattern(with_bias=True))
    conv2d_relu_pat = ("dnnl.conv2d_relu", make_pattern(with_bias=False))
    dnnl_patterns = [conv2d_bias_relu_pat, conv2d_relu_pat]
    return dnnl_patterns

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
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
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

#mod = tvm.IRModule()
#mod["main"] = tvm.relay.Function([x], y)
mod, params, data_shape, out_shape = get_network(model_name, batch_size)

mod = tvm.relay.transform.MergeComposite(pattern_table())(mod)
mod = tvm.relay.transform.AnnotateTarget(["dnnl"])(mod)  # Output: Figure 2
mod = tvm.relay.transform.MergeCompilerRegions()(mod)  # Output: Figure 3
mod = tvm.relay.transform.PartitionGraph()(mod)  # Output: Figure 4

graph, module, params = tvm.relay.build(mod, target="llvm")

print(graph)
#print(module)
print(params)
