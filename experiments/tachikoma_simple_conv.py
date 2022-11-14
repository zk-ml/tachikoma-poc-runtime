import numpy as np
import tvm
from tvm.relay.op.contrib import tachikoma
from tvm import relay
from utils import partition_for_tachikoma

dtype = "int8"
scale = 100
dshape = (64, 1, 32, 32)
kshape = (1, 1, 1, 1)

device = tvm.cpu()
target = "llvm"

data = np.random.uniform(-scale, scale, size=dshape).astype(dtype)
kern = np.random.uniform(-scale, scale, size=kshape).astype(dtype)

w = relay.var("weight", shape=kshape, dtype=dtype)
x = relay.var("x", shape=dshape, dtype=dtype)
y = relay.nn.conv2d(
    x, w, padding=(1, 1), dilation=(1, 1), groups=1, channels=1, kernel_size=(1, 1)
)
z = relay.nn.relu(y)

_mod = tvm.IRModule()
_mod["main"] = relay.Function([x, w], z)

params = {"weight": kern, "x": data}
mod = partition_for_tachikoma(_mod, params)
print(mod["main"].astext(show_meta_data=False), "\n")
print(mod.get_global_vars())

with tvm.transform.PassContext(opt_level=1):
    func = relay.create_executor("graph", mod=mod, device=device, target=target).evaluate()

with tvm.transform.PassContext(opt_level=1):
    func_ref = relay.create_executor("graph", mod=_mod, device=device, target=target).evaluate()

print(params.keys())

for i in range(3):
    pred = func(**params)
    actual = func_ref(**params)
    err = (pred.numpy() - actual.numpy()).mean()
    print(f"iter {i}: err {err}")
    # print(pred.numpy()[0,0,0,:5])
    # print(actual.numpy()[0,0,0,:5])
