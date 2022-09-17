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
mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype="float32")
print(mod["main"].astext(show_meta_data=False), "\n")
mod = tachikoma.partition_for_tachikoma(mod, params)
print(mod["main"].astext(show_meta_data=False), "\n")
print(mod.get_global_vars())
print(type(mod))

with tvm.transform.PassContext(opt_level=1):
    lib, bldmod = build_module.build_with_bldmod(mod, target=target, params=params)

path_set = tvm.get_global_func("runtime.TachikomaSetExportPath")

explib = bldmod._get_module()
rmod = lib["default"](device)
#print(type(explib))
#print(type(lib["default"]))
print(lib)
print(explib)
print(type(explib))

rt_mod = tvm.contrib.graph_executor.GraphModule(rmod)

print("subsequent runs")
for i in range(2):
    path_set(explib, f"/data/tachikoma_results/serialized_{i}.ndarray")
    
    for name, data in lib.get_params().items():
        print(name, data.shape)
        data = tvm.nd.array(data.numpy() + i)
        rt_mod.set_input(name, data)
    rt_mod.run()

    out = rt_mod.get_output(0)

