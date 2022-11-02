# tachikoma: open interface between TensorIR and arithmetic circuits

On-chain deep learning is prohibitive because blockchains are not designed to run neural networks (and they shouldn't). The structure and sheer # of computations of modern architectures necessitate specialized hardware & instruction sets to squeeze as much performance as possible. Convolutional neural networks, for example, were barely effective before the advent of CUDA and only gained popularity with a thriving developer ecosystem full of mature tools. What consensus can instead offer is to prove certain characteristics of such networks, which motivates "zero-knowledge machine learning." 

The idea of certifying properties like performance metrics on NNs without revealing the underlying data or network parameters feels more enticing, and encoding+running only the verifier in a given proof system is often much cheaper than the inference computation itself. Several POCs demonstrate this idea, including something I wrote a while ago around a simple linear regression, but ultimately the current paradigm of handcrafting circuits for exact network configuration is not scalable. 

Neural network operators and architectures evolve drastically, and it is unlikely that machine learning teams will bear the cost (talent & resource) to re-implement their production pipelines in DSLs/libraries that are "zero-knowledge friendly." Additionally, it is unclear which zk proof system will eventually win as they all need orders of magnitude of improvement in performance and developer experience to reach wide adoption. Writing a neural network library in one of them seems to take an unnecessary risk. For example, one may spend a nontrivial time encoding a particular 12-layer quantized BERT base in snarkjs. Still, that effort translates very poorly if snarkjs loses out on a 2-year horizon or if the transformer implementation evolves as well (linear attention, swapping layers for MLP, etc.)

We need a compiler from production-grade neural network frameworks to a platform that is friendly to a wide range of zk proof systems. As of now, the answer seems to be the TVM compiler stack:
* ML teams implement neural networks in popular, performance-oriented frameworks -> TVM compiles models from a wide range of frameworks (PyTorch, Tensorflow, ONNX) into TensorIR.
* Many neural networks go through optimizations before deployment, so the structure you are proving may be different than the structure you've trained on -> TVM's TensorIR implementation has optimization built-in.
* Most zero-knowledge-proof systems do not support float arithmetic -> TVM comes with both automatic quantization and compilation of prequantized models.
* Proving a neural network's property requires special setups in proof systems that most NN inference engines cannot handle -> TVM is very friendly to the concept of bring-your-own-codegen (BYOC), such that developers of specialized hardware and "exotic" runtime environments (like zk) can solely focus on implementing individual operators, instead of worrying about "gluing" runtime code back into the compiler, or other necessary optimizations.

I've been working on a project, tachikoma, that seeks to extend TVM into a runtime for zero-knowledge machine learning use cases. This will involve two parts: 

1. The tachikoma runtime, which performs code generation from TensorIR to an arbitrary zero-knowledge proof system. It will be very similar to the Intel DNNL (oneAPI Deep Neural Network Library) JSON runtime as part of the BYOC offering. Tachikoma compiles & runs a neural network (an optionally optimized graph in TensorIR ported from popular frameworks like PyTorch) via TVM's GraphExecutor while exposing a language-agnostic RPC API that specifies the input, output, weights, and other configurations for each operator. This way, an arbitrary zero-knowledge proof builder could attach to the runtime to gradually build back the original computational graph while proving it. 

2. As an example, a rust zero-knowledge NN prover, tentatively based on plonky2 and written in Rust. It implements the tachikoma client-side RPC API and ideally produces proofs of productionizable neural networks comprising the most popular operators (convolutions, fully-connected, attention, etc.) 

This will be a fully open-sourced and "wish I had more free time" project, so I will sporadically work on it and publish all progress here: https://github.com/zk-ml.

Finally, thanks to the people I've consulted around this idea: @szgbo, @junrushao, @recmo, @anihamde, @bhuan_, @lm_zheng, @LigengZhu, @YassineLanda, and @l_udens. Excited for things to come!

# Citing

If you find this helpful, please consider citing:

```
@misc{liao_tachikoma_2022, 
    title={tachikoma: open interface between TensorIR and arithmetic circuits}, url={https://github.com/zk-ml/tachikoma-runtime},
    author={Liao, Peiyuan},
    year={2022}
} 
```
