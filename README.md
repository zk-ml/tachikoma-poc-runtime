<img src=https://raw.githubusercontent.com/zk-ml/linear-a-site/main/logo/linear-a-logo.png width=64/> tachikoma: neural network inference standard for arithmetic circuits


---------------

tachikoma is an open standard for neural network inference bridging Apache TVM to arithmetic circuits in zero-knowledge-proof systems. 

tachikoma defines how a neural network's inference process should be serialized into a graph of operator computational traces, each of which containing the input, expected output, relevant metadata (including parameters), and an identifier relating back to the original operator in TVM's intermediate representation.

This is a proof of concept showing how tachikoma can be used in ZKP systems. We will be implementing a simple graph runtime on top of the tachikoma standard, as well as a circuit builder in ZEXE.
