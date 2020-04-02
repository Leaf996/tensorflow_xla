# What tensorflow xla is ?
- XLA(Accelerated Linear Algebra) is developed by google, a ***complier*** for Tensorflow.
# Why we need tensorflow xla ?
- One of the design goals and core strengths of Tensorflow is its ***flexibility***. Tensorflow was designed to be a flexible and extensible system for defining arbitrary data flow graphs and executing then efficiently in a distributed manner using heterogenous computing devices(such as CPUs and GPUs).
- But flexibility is often at odds with performance. While Tensorflow aims to let you define any kind of data flow graph, it's challenging to make all graphs execute efficiently because Tensorflow optimizes each op separately. When an op with an efficient implementation exists or when each op is a relatively heavyweight operation, all is well; otherwise, the user can still compose this op out of lower-level ops, but this composition is not guaranteed to run in the most efficient way.
# What tensorflow xla does ?
- XLA uses ***JIT(just-in-time)*** complication techniques to analyze the Tensorflow graph created by the user at runtime, specialize it for the actual runtime **dimensions** and **types**, **fuse multiple ops together** and **emit efficient native machine code** for them - for devices like CPUs, GPUs and custome accelerators(e.g. Google's TPU).
# What tensorflow xla improved ?
- **Improve execution speed**. Compile subgraphs to reduce the execution time of short-lived Ops to eliminate overhead from the Tensorflow runtime, fuse pipelined operations to reduce memory overhead, and specialize to known tensor shapes to allow for more aggressive constant propagation.
- **Improve memory usage**. Analyze and schedule memory usage, in principle eliminate many intermediate storage buffers.
- **Reduce reliance on custom Ops**. Remove the need for many custom Ops by improving the performance of automatically fused low-level Ops to match the preformance of custom Ops that were fused by hand.
- **Reduce mobile footprint**. Eliminate the Tensorflow runtime by ***ahead-of-time*** compiling the subgraph and emitting an object/header file pair that can be linked directly into another application. The results can reduce the footprint for mobile inference by several orders of magitude.
- **Improve protability**. Make it relatively easy to write a new backend for novel hardware, at which point a large fraction of Tensorflow programs will run unmodified on that hardware. This is in contrast with the approach of specializing individual monolithic Ops for new hardware, which requires Tehsorflow programs to be rewritten to make use of those Ops.
# How does XLA work ?
- TODO
# Reference
- [XLA-TensorFlow, compiled][1]
- [XLA Architecture][2]

[1]:[https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html]
[2]:https://www.tensorflow.org/xla/architecture