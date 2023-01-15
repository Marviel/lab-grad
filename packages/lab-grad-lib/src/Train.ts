import _ from "lodash";
import { MLP } from "./MLP";
import { createSimpleLogger } from "./utils/logging";
import { newVal, Value } from "./Value";

interface TrainerOptions<TInput extends Value[], TOutput extends Value[]> {
    // A set of training samples.
    // Each sample is an object with an input and an output.
    samples: { input: [...TInput]; groundTruthOutput: [...TOutput] }[];

    // A function which will take in a set of input values and return a set of output values.
    // The values returned should always be chained to the input values, so that the gradient
    // can be computed.
    predict: (input: [...TInput]) => [...TOutput];

    // Get Parameters of the network.
    // This is used to clear the gradients of the network,
    // and to update the network parameters based on the gradient 
    // produced by backpropagating the loss function.
    getParameters: () => Value[];

    // Computes a single loss Value from a single prediction.
    computeLoss(groundTruthOutput: [...TOutput], predictedOutput: [...TOutput]): Value;

    learningRate?: number;

    // Sample splitting.
    // If provided, will split the samples into a training set and a validation set.
    // This helps prevent overfitting.
    sampleSplitRatio?: number;

    // The maximum number of epochs to run. Defaults to 1.
    maxEpochs?: number;

    // Samples per epoch. Defaults to the number of samples.
    samplesPerEpoch?: number;
}

interface TrainerSampleResult {
    // The loss for this sample.
    lossData: number;
    // The predicted outputs for this sample.
    predictedOutput: number[];
    // The sample itself.
    sample: { input: Value[]; groundTruthOutput: Value[] };
}

export function trainer<TIn extends Value[], TOut extends Value[]>(opts: TrainerOptions<TIn, TOut>) {
    const logger = createSimpleLogger('train', false);

    const params = opts.getParameters();
    const sampleResults: TrainerSampleResult[] = []

    const learningRate = opts.learningRate || 0.1;

    _.forEach(opts.samples, (sample) => {
        // Forward pass.
        // Run the inputs through the model to get our prediction.
        const predictedOutput = opts.predict(sample.input);

        // Determine the loss for this sample.
        // This is the difference between the predicted output and the ground truth output.
        // High loss is bad.
        const loss = opts.computeLoss(sample.groundTruthOutput, predictedOutput)

        // Clear the gradients for our parameters.
        params.forEach((p) => p.grads.clear())

        // Backpropagate the loss to get the gradient for each parameter.
        loss.backward();

        // Update the model parameters using the gradients.
        // Basically -- how did this parameter affect the loss?
        // We adjust this parameter slightly such that the loss should be reduced.
        params.forEach((p) => p.data += -learningRate * p.grads.get(loss.uniqId));

        sampleResults.push({
            lossData: loss.data,
            predictedOutput: predictedOutput.map((p) => p.data),
            sample,
        });
    })

    return sampleResults;
}



function array<X extends unknown[]>(x: [...X]): [...X] {
    return [1, 2, 3] as any
}


const ret = array([new Value(1), new Value(2), new Value(3), new Value(4)])