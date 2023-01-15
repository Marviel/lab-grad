// class Layer:

import _ from "lodash";
import { Neuron, NeuronOptions } from "./Neuron";
import { Value } from "./Value";

export interface LayerOptions {
    neuronOptions: Partial<NeuronOptions>;
}

function getDefaultLayerOptions(): LayerOptions {
    return {
        // No special neuron options.
        neuronOptions: {},
    }
}

export class Layer {
    neurons: Neuron[];
    _opts: LayerOptions = getDefaultLayerOptions();
    constructor(numInputs: number, numOutputs: number, opts: Partial<LayerOptions> = {}) {
        // Get options merged with defaults.
        this._opts = _.defaultsDeep({}, opts, this._opts);
        this.neurons = _.range(numOutputs).map(() => new Neuron(numInputs, this._opts.neuronOptions));
    }

    run(inputs: Value[]): Value[] {
        const outs = this.neurons.map((n) => n.run(inputs));

        return outs;
    }

    get parameters(): Value[] {
        return _.flatten(this.neurons.map((n) => n.parameters));
    }
}