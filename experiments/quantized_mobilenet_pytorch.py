from tvm import testing

testing.utils.install_request_hook(depth=3)
from PIL import Image

import numpy as np

import torch
from torchvision.models import quantization

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata


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


def get_synset():
    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        return eval(f.read())


def run_tvm_model(mod, params, input_name, inp, target="llvm"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_executor.GraphModule(
        lib["default"](tvm.device(target, 0))
    )

    runtime.set_input(input_name, inp)
    runtime.run()
    return runtime.get_output(0).numpy(), runtime


synset = get_synset()

inp = get_imagenet_input()


def quantize_model(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration
    model(inp)
    torch.quantization.convert(model, inplace=True)


qmodel = quantization.resnet18(pretrained=True, quantize=True).eval()

pt_inp = torch.from_numpy(inp)
# quantize_model(qmodel, pt_inp)
script_module = torch.jit.trace(qmodel, pt_inp).eval()

with torch.no_grad():
    pt_result = script_module(pt_inp).numpy()

print(script_module)

input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(
    script_module, input_shapes, keep_quantized_weight=True
)

# Tachikoma
from tvm.relay.op.contrib.tachikoma import make_qnn_conv2d_pattern
from tvm.relay.op.contrib.register import register_pattern_table


@register_pattern_table("tachikoma")
def pattern_table():
    dnnl_patterns = [make_qnn_conv2d_pattern()]
    return dnnl_patterns


patterns = pattern_table()
# print(patterns)
mod = relay.transform.MergeComposite(patterns)(mod)
mod = relay.transform.AnnotateTarget(["tachikoma"])(mod)
mod = relay.transform.MergeCompilerRegions()(mod)
mod = relay.transform.PartitionGraph()(mod)

print(mod)

target = tvm.target.Target("llvm", host="llvm")

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

from tvm.contrib import graph_executor

rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
rt_mod.set_input("input", inp)
rt_mod.run()
tvm_result = rt_mod.get_output(0).numpy()

print(np.absolute(tvm_result - pt_result).mean())
