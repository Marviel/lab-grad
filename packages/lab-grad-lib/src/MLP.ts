// class MLP:

import _ from "lodash";
import { Layer } from "./Layer";
import { Value } from "./Value";


//   def __init__(self, nin, nouts):
//     sz = [nin] + nouts
//     self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

//   def __call__(self, x):
//     for layer in self.layers:
//       x = layer(x)
//     return x

//   def parameters(self):
//     return [p for layer in self.layers for p in layer.parameters()]

export class MLP {
    layers: Layer[];
    constructor(numInputs: number, numOutputs: number[]) {
        const sizes = [numInputs, ...numOutputs];
        this.layers = sizes.slice(0, -1).map((numInputs, idx) => new Layer(numInputs, sizes[idx + 1]));
    }

    /**
     * Pass the inputs into this network and return the output as a Value.
     * @param inputs The inputs as Value objects.
     * @returns 
     */
    run(inputs: Value[]): Value[] {
        const res = this.layers.reduce((acc, layer) => layer.run(acc), inputs);

        return this.layers.reduce((acc, layer) => layer.run(acc), inputs);
    }

    /**
     * Get all parameters in the network.
     */
    get parameters(): Value[] {
        return _.flatten(this.layers.map((l) => l.parameters));
    }
}