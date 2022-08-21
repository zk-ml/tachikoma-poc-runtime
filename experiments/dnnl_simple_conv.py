import numpy
import tvm
import tvm.relay
from tvm.relay.dataflow_pattern import is_op, wildcard
from tvm.relay.op.contrib.register import register_pattern_table


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(attrs, args):
        return supported

    return _func_wrapper


_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.relu")
_register_external_op_helper("add")
_register_external_op_helper("subtract")
_register_external_op_helper("multiply")


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
mod["main"] = tvm.relay.Function([x], y)

mod = tvm.relay.transform.MergeComposite(pattern_table)(mod)
mod = tvm.relay.transform.AnnotateTarget(["dnnl"])(mod)  # Output: Figure 2
mod = tvm.relay.transform.MergeCompilerRegions()(mod)  # Output: Figure 3
mod = tvm.relay.transform.PartitionGraph()(mod)  # Output: Figure 4

with tvm.transform.PassContext(opt_level=3):
    graph, module, params = tvm.relay.build(mod, target="llvm")

print(graph)

