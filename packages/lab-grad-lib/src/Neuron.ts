import _ from "lodash";
import { Value } from "./Value";

export interface NeuronOptions {
    // This is a function that takes a Value and returns a Value.
    // It is generally used to apply a nonlinearity to the output of a neuron.
    // The default value is `tanh`.
    applyNonlinearity: (x: Value) => Value;
}

export function getDefaultNeuronOptions(): NeuronOptions {
    return {
        applyNonlinearity: (x) => x.tanh(),
    }
}

/**
 * A single neuron.
 * 
 * @param numInputs The number of inputs to this neuron.
 * @param opts Options for this neuron.
 */
export class Neuron {
    bias: Value;
    weights: Value[];
    _opts: NeuronOptions = getDefaultNeuronOptions();
    constructor(numInputs: number, opts?: Partial<NeuronOptions>) {
        // Get options merged with defaults.
        this._opts = _.defaultsDeep({}, opts, this._opts);

        // Create our neuron parameters, randomly initialized between -1 and 1.
        this.bias = new Value(_.random(-1, 1));
        this.weights = _.range(numInputs).map(() => new Value(_.random(-1, 1)));
    }

    run(inputs: Value[]) {
        if (inputs.length !== this.weights.length) {
            throw new Error(`Expected ${this.weights.length} inputs, but got ${inputs.length}`);
        }

        const weightedInputs = inputs.map((input, idx) => input.mult(this.weights[idx]));
        const sum = weightedInputs.reduce((acc, val) => acc.add(val));
        const output = this._opts.applyNonlinearity(sum.add(this.bias))
        return output;
    }

    get parameters() {
        return [this.bias, ...this.weights];
    }
}