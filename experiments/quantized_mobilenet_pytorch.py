from tvm import testing

testing.utils.install_request_hook(depth=3)
from PIL import Image

import numpy as np

import torch
from torchvision.models import quantization

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import tachikoma
from tvm.relay import transform


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
with TempOpAttr("nn.conv2d", "FTVMLegalize", tachikoma.legalize_group_conv):
    with TempOpAttr(
        "nn.conv2d_transpose", "FTVMLegalize", tachikoma.legalize_group_conv
    ):
        seq = tvm.transform.Sequential(
            [
                transform.CanonicalizeOps(),
                transform.InferType(),
                transform.SimplifyInference(),
                transform.FoldConstant(),
                transform.FoldScaleAxis(),
                # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
                transform.SimplifyExpr(),
                transform.FoldConstant(),
                # alter group conv /conv_transpose layout to `GOIHW` / `GIOHW`
                transform.Legalize(),
                transform.FoldConstant(),
            ]
        )
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)
alter_layout = True
if alter_layout:
    with TempOpAttr("nn.conv1d", "FTVMAlterOpLayout", tachikoma.alter_conv):
        with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", tachikoma.alter_conv):
            with TempOpAttr("nn.conv3d", "FTVMAlterOpLayout", tachikoma.alter_conv):
                with TempOpAttr(
                    "nn.conv2d_transpose",
                    "FTVMAlterOpLayout",
                    tachikoma.alter_conv_transpose,
                ):
                    with TempOpAttr(
                        "nn.conv3d_transpose",
                        "FTVMAlterOpLayout",
                        tachikoma.alter_conv_transpose,
                    ):
                        alter_layout_seq = tvm.transform.Sequential(
                            [
                                transform.AlterOpLayout(),
                                transform.FoldConstant(),
                            ]
                        )
                        with tvm.transform.PassContext(opt_level=3):
                            mod = alter_layout_seq(mod)

mod = tachikoma.rewrite_layer_norm(mod)
mod = tachikoma.rewrite_dense_bias_gelu_reshape_last(mod)
mod = tachikoma.legalize_qnn_for_tachikoma(mod)

byoc_seq = tvm.transform.Sequential(
    [
        transform.MergeComposite(tachikoma.pattern_table()),
        transform.AnnotateTarget("tachikoma"),
        transform.MergeCompilerRegions(),
        transform.PartitionGraph(),
    ]
)
target = tvm.target.Target("llvm", host="llvm")

with tvm.transform.PassContext(opt_level=3):
    mod = byoc_seq(mod)
    lib = relay.build(mod, target=target, params=params)

from tvm.contrib import graph_executor

rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
rt_mod.set_input("input", inp)
rt_mod.run()
tvm_result = rt_mod.get_output(0).numpy()

print(np.absolute(tvm_result - pt_result).mean())
