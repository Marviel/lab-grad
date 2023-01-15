// class MLP:

import _ from "lodash";
import { Layer } from "./Layer";
import { Value } from "./Value";

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