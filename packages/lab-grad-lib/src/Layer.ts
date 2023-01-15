// class Layer:

import _ from "lodash";
import { Neuron, NeuronOptions } from "./Neuron";
import { Value } from "./Value";


//   def __init__(self, nin, nout):
//     self.neurons = [Neuron(nin) for _ in range(nout)]

//   def __call__(self, x):
//     outs = [n(x) for n in self.neurons]
//     return outs[0] if len(outs) == 1 else outs

//   def parameters(self):
//     return [p for neuron in self.neurons for p in neuron.parameters()]

export interface LayerOptions {
    neuronOptions: Partial<NeuronOptions>;
}

export class Layer {
    neurons: Neuron[];
    constructor(numInputs: number, numOutputs: number, readonly applyNonlinearity: (x: Value) => Value = (x) => x.tanh()) {
        this.neurons = _.range(numOutputs).map(() => new Neuron(numInputs, { applyNonlinearity }));
    }

    run(inputs: Value[]): Value[] {
        const outs = this.neurons.map((n) => n.run(inputs));

        return outs;
    }

    get parameters(): Value[] {
        return _.flatten(this.neurons.map((n) => n.parameters));
    }
}