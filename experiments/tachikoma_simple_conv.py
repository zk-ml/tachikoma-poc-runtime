import numpy as np
import tvm
from tvm.relay.op.contrib import tachikoma
from tvm import relay

dtype = "float32"
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

lib = mod.libmod
export_fn = tvm.get_global_func("runtime.TachikomaExportModule")

print("first run")
export_fn(lib, "/data/tachikoma_results/serialized.ndarray")

device = tvm.cpu()
target = "llvm"

with tvm.transform.PassContext(opt_level=3):
    rt_mod = relay.create_executor(
        "graph", mod=mod, device=device, target=target
    )
    func = rt_mod.evaluate()

print("subsequent runs")
for i in range(5):
    inp = {}
    for name, data in params.items():
        print(name, data.shape)
        data = tvm.nd.array(data.numpy() + i)
        inp[name] = data
    out = func(**inp)

    export_fn(lib, f"/data/tachikoma_results/serialized_{i}.ndarray")
