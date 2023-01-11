# Lab-Grad
This is a (WIP :) typescript implemenation of a Pytorch-style autograd library, along with various tools for playing with the associated algorithms.

The underlying Autograd library, `packages/lab-grad-lib`, is directly inspired by [Andrej Karpathy's excellent "Micrograd" library](https://github.com/karpathy/micrograd).

### Main Files
- [packages/lab-grad-lib/src/Value.ts](https://github.com/Marviel/lab-grad/blob/main/packages/lab-grad-lib/src/Value.ts): The core Autograd Engine

### Differences from Micrograd
- Gradients for a given node can be calculated with respect to multiple nodes at once. I don't currently see an immediate use for this, but it made more sense for me to build it in, as it wasn't always clear what `grad` meant in the original implementation.
