# Lab-Grad
This is a (WIP :) Typescript implemenation of a Pytorch-style autograd library, along with various tools for playing with the associated algorithms.

Because it is in Typescript, it should run in your browser.


>  For now, Lab-Grad is only intended for educational purposes :) If you want to run neural networks in your browser in a more production-friendly way, check out ONNX.


The underlying Autograd library, `packages/lab-grad-lib`, is directly inspired by [Andrej Karpathy's excellent "Micrograd" library](https://github.com/karpathy/micrograd).

### Main Files
- [packages/lab-grad-lib/src/Value.ts](https://github.com/Marviel/lab-grad/blob/main/packages/lab-grad-lib/src/Value.ts): The core Autograd Engine

### 🔍 Differences from Micrograd
- Types! :)
- Gradients for a given node can be calculated with respect to multiple nodes at once. I don't currently see an immediate use for this, but it made more sense for me to build it in, as it wasn't always clear what `grad` meant in the original implementation.
- As I work, I will be adding visualizations, and documentation.


### 🔄 TODO
#### Now
- [x] Implement the `Value` object from Micrograd -- i.e. non-tensor math
- [x] Implement basic Webpage for In-Browser Neuron visualization
- [ ] Implement simple classification training loop using Value object -- with visual I/O
- [ ] Visualize neuron gradients during learning task
- [ ] Blogpost explaining progress
- [ ] Ship webpage
#### Next
- [ ] Implement Tensor math
- [ ] Implement Tensor visualization
#### Later
- [ ] Implement Transformers
- [ ] WASM or Rust bindings for Matrix Math optimization
- [ ] Implement Semi-GPT-2
- [ ] Implement Stable Diffusion
