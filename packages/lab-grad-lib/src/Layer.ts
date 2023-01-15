// class Layer:

import _ from "lodash";
import { Neuron, NeuronOptions } from "./Neuron";
import { Value } from "./Value";

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