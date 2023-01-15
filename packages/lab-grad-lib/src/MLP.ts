// class MLP:

import _ from "lodash";
import { Layer, LayerOptions } from "./Layer";
import { Value } from "./Value";

interface MLPOptions {
    layerOptions: Partial<LayerOptions>;
    individualLayerOptions: Partial<LayerOptions | undefined>[];
}

function getDefaultMLPOptions(): MLPOptions {
    return {
        // No special layer options.
        layerOptions: {},
        // No special individual layer options.
        individualLayerOptions: [],
    }
}

/**
 * A multi-layer perceptron.
 * 
 * @param numInputs The number of inputs to the network.
 * @param numOutputs The number of outputs from the network.
 */
export class MLP {
    layers: Layer[];
    _opts: MLPOptions = getDefaultMLPOptions();
    constructor(numInputs: number, numOutputs: number[], opts?: Partial<MLPOptions>) {
        // Get options merged with defaults.
        this._opts = _.defaultsDeep({}, opts, this._opts);

        const sizes = [numInputs, ...numOutputs];
        this.layers = sizes.slice(0, -1).map((numInputs, idx) =>
            new Layer(
                numInputs, sizes[idx + 1],
                // Either our individual layer options or our default layer options, if none were provided.
                this._opts.individualLayerOptions[idx] || this._opts.layerOptions
            )
        );
    }

    /**
     * The forward pass for this network.
     * 
     * Pass the inputs into this network and return the output of the forward pass as a Value.
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