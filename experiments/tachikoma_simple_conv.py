import numpy as np
import tvm
from tvm.relay.op.contrib import tachikoma
from tvm import relay

dtype = "int8"
scale = 100
dshape = (64, 1, 32, 32)
kshape = (1, 1, 1, 1)

scale = 100

data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
kern = np.random.uniform(-scale, scale, size=kshape).astype(dtype)

w = relay.var("weight", shape=kshape, dtype=dtype)
x = relay.var("x", shape=dshape, dtype=dtype)
y = relay.nn.conv2d(
    x, w, padding=(1, 1), dilation=(1, 1), groups=1, channels=1, kernel_size=(1, 1)
)
z = relay.nn.relu(y)

mod = tvm.IRModule()
mod["main"] = relay.Function([x, w], z)

params = {"weight": kern, "x": data}
mod = tachikoma.partition_for_tachikoma(mod, params)
print(mod["main"].astext(show_meta_data=False), "\n")
print(mod.get_global_vars())

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="llvm", params=params)

path_set = tvm.get_global_func("runtime.TachikomaSetExportPath")

explib = lib.get_lib()

device = tvm.cpu()
rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

print("subsequent runs")
for i in range(5):
    for name, data in lib.get_params().items():
        print(name, data.shape)
        data = tvm.nd.array(data.numpy() + i)
        rt_mod.set_input(name, data)
    rt_mod.run()

    out = rt_mod.get_output(0)

    path_set(explib, f"/data/tachikoma_results/serialized_{i}.ndarray")
