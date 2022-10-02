import numpy as np
import tvm
from tvm.relay.op.contrib import tachikoma
from tvm import relay
from tvm.relay import build_module
import tvm.relay.testing

device = tvm.cpu()
target = "llvm"
dtype = "float32"
ishape = (1, 3, 224, 224)
mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype=dtype)
print(mod["main"].astext(show_meta_data=False), "\n")
mod = tachikoma.partition_for_tachikoma(mod, params)
print(mod["main"].astext(show_meta_data=False), "\n")
print(mod.get_global_vars())
print(type(mod))

with tvm.transform.PassContext(opt_level=1):
    func = relay.create_executor("graph", mod=mod, device=device, target=target).evaluate()

print(params.keys())

for _ in range(3):
    input_dict = {"data": np.random.uniform(*ishape).astype("float32")}
    func(**input_dict, **params)
