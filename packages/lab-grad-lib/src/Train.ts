import _ from "lodash";
import { MLP } from "./MLP";
import { createSimpleLogger } from "./utils/logging";
import { newVal, Value } from "./Value";

export interface TrainerOptions<TInput extends Value[], TOutput extends Value[]> {
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

    // If provided, this will be called after each sample is processed.
    sampleCompleteCallback?: (sampleResult: TrainerSampleResult & { loss: Value }) => void;

    // Sample splitting.
    // If provided, will split the samples into a training set and a validation set.
    // This helps prevent overfitting.
    sampleSplitRatio?: number;

    // The maximum number of epochs to run. Defaults to 1.
    maxEpochs?: number;

    // Samples per epoch. Defaults to the number of samples.
    samplesPerEpoch?: number;

    stepTime?: number;
}

export interface TrainerSampleResult {
    // The loss for this sample.
    lossData: number;
    // The predicted outputs for this sample.
    predictedOutput: number[];
    // The sample itself.
    sample: { input: Value[]; groundTruthOutput: Value[] };
}

async function resolvePromisesSeq<T>(tasks: Promise<T>[]) {
    const results = [];
    for (const task of tasks) {
        results.push(await task);
    }

    return results;
};

function delay(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}


export class Trainer {
    private stepIndex: number = 0;
    private params: Value[] = [];
    constructor(private opts: TrainerOptions<any, any>) {
        this.params = opts.getParameters();
    }

    getNextSample() {
        return this.opts.samples[this.stepIndex];
    }

    // Runs the next sample, and updates the index.
    runNextStep() {
        const { opts } = this;
        const { learningRate = 0.1 } = opts;
        const sample = this.getNextSample()

        this.runSample(sample);

        this.stepIndex += 1;
    }

    // Runs a sample.
    // Does NOT update the step index.
    runSample(sample: { input: Value[]; groundTruthOutput: Value[] }) {
        const { opts } = this;
        const { learningRate = 0.1 } = opts;

        // Forward pass.
        // Run the inputs through the model to get our prediction.
        const predictedOutput = this.opts.predict(sample.input);

        // Determine the loss for this sample.
        // This is the difference between the predicted output and the ground truth output.
        // High loss is bad.
        const loss = opts.computeLoss(sample.groundTruthOutput, predictedOutput)

        // Clear the gradients for our parameters.
        this.params.forEach((p) => p.grads.clear())

        // Backpropagate the loss to get the gradient for each parameter.
        loss.backward();

        // Update the model parameters using the gradients.
        // Basically -- how did this parameter affect the loss?
        // We adjust this parameter slightly such that the loss should be reduced.
        this.params.forEach((p) => p.data += -learningRate * p.grads.get(loss.uniqId));

        const sampleResult = {
            lossData: loss.data,
            predictedOutput: predictedOutput.map((p) => p.data),
            sample,
        }

        // Call our callback.
        opts.sampleCompleteCallback && opts.sampleCompleteCallback({ ...sampleResult, loss });

        return sampleResult;
    }

    runAllSteps() {
        return this.opts.samples.map((sample) => {
            return this.runSample(sample);
        })
    }
}



// TODO make it to where we can run a single cycle of trainer at once. Probably requires a Trainer object.

export async function trainer<TIn extends Value[], TOut extends Value[]>(opts: TrainerOptions<TIn, TOut>): Promise<TrainerSampleResult[]> {
    const logger = createSimpleLogger('train', false);

    const params = opts.getParameters();

    const learningRate = opts.learningRate || 0.1;

    const trainer = new Trainer(opts);

    return trainer.runAllSteps();
}