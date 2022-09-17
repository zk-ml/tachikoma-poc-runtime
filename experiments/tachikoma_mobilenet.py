import numpy as np
import tvm
from tvm.relay.op.contrib import tachikoma
from tvm import relay
from tvm.relay import transform
import tvm.relay.testing

path_set = tvm.get_global_func("runtime.TachikomaSetExportPath")

dtype = "float32"
ishape = (1, 3, 224, 224)
mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype="float32")
#print(mod["main"].astext(show_meta_data=False), "\n")
#mod = transform.AnnotateTarget(["tachikoma"])(mod)
#mod = transform.MergeCompilerRegions()(mod)
#mod = transform.PartitionGraph()(mod)
mod = tachikoma.partition_for_tachikoma(mod, params)
print(mod["main"].astext(show_meta_data=False), "\n")
print(mod.get_global_vars())
print(type(mod))

#with tvm.transform.PassContext(opt_level=1):
#    lib = relay.build(mod, target="llvm", params=params)
#rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](device))
#explib = lib.get_lib()
#print(type(explib))
#print(type(lib))
#print(explib)

device = tvm.cpu()
target = "llvm"

#print("subsequent runs")
for i in range(2):
    #path_set(explib, f"/data/tachikoma_results/serialized_{i}.ndarray")

    with tvm.transform.PassContext(opt_level=3):
        func = relay.create_executor("graph",
            mod=mod, device=device, target=target).evaluate()
    
    #func(**params)
    
