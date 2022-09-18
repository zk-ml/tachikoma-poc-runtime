import numpy as np
import tvm
from tvm.relay.op.contrib import tachikoma
from tvm import relay
from tvm.relay import build_module
import tvm.relay.testing
import torch
from torchvision.models import quantization
from tvm.contrib.download import download_testdata
from PIL import Image

def get_transform():
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

def get_real_image(im_height, im_width):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    return Image.open(img_path).resize((im_height, im_width))

def get_imagenet_input():
    im = get_real_image(224, 224)
    preprocess = get_transform()
    pt_tensor = preprocess(im)
    return np.expand_dims(pt_tensor.numpy(), 0)

inp = get_imagenet_input()
qmodel = quantization.resnet18(pretrained=True, quantize=True).eval()
pt_inp = torch.from_numpy(inp)
script_module = torch.jit.trace(qmodel, pt_inp).eval()

with torch.no_grad():
    pt_result = script_module(pt_inp).numpy()

print(script_module)

input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(
    script_module, input_shapes, keep_quantized_weight=True
)

device = tvm.cpu()
target = "llvm"
print(mod["main"].astext(show_meta_data=False), "\n")
mod = tachikoma.partition_for_tachikoma(mod, params)
print(mod["main"].astext(show_meta_data=False), "\n")
print(mod.get_global_vars())
print(type(mod))

with tvm.transform.PassContext(opt_level=1):
    lib, bldmod = build_module.build_with_bldmod(mod, target=target, params=params)

rmod = lib["default"](device)
rt_mod = tvm.contrib.graph_executor.GraphModule(rmod)

for i in range(2):    
    for name, data in lib.get_params().items():
        print(name, data.shape)
        data = tvm.nd.array(data.numpy() + i)
        rt_mod.set_input(name, data)
    rt_mod.run()

    out = rt_mod.get_output(0)