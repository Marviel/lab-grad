import _, { random } from 'lodash';
import { newVal, Value } from '../../src/Value'
// import console from 'console';
import { Neuron } from '../../src/Neuron';
import { average } from '../../src/utils/Math';
import { MLP } from '../../src/MLP';
import { createSimpleLogger } from '../../src/utils/logging';
import { trainer } from '../../src/Train';

function randomArray(length: number, low: number, high: number) {
    return _.range(0, length).map(() => _.random(low, high));
}

test('can learn XOR', () => {
    const logger = createSimpleLogger('can learn XOR', false);
    // Our target function is a simple XOR function.
    const targetFunction = (x: number, z: number) => +(x != z);

    const mlp = new MLP(2, [2, 2]);

    // All the params of the whole network.
    const params = mlp.parameters;

    // Create the model.
    // Create a simple neural network with two inputs, one hidden layer with two nodes, and one output node.
    const m = (x: number, z: number) => {
        const in_a = newVal(x);
        const in_b = newVal(z);
        // Run the inputs through the hidden layer.
        const mlp_outs = mlp.run([in_a, in_b]);

        // Combine the outputs of the hidden layer via addition.
        const out = mlp_outs.reduce((acc, hn) => acc.add(hn))
        return out;
    }

    const losses: any[] = []


    // Train the model.
    _.range(0, 1000).forEach((idx) => {
        // console.log(`--- Episode ${idx} ---------------------------------------------`)
        // Create a random input.
        const [x, z] = randomArray(2, 0, 1);

        // Get the groundTruth.
        const ygt = targetFunction(x, z);

        // Run the random inputs through the model to get our prediction.
        const ypred = m(x, z);

        // Calculate the squared loss.
        const loss = (ypred.sub(ygt).pow(2))

        // Clear the gradients for our parameters.
        // Backpropagate the loss to get the gradient for each parameter.
        params.forEach((p) => p.grads.clear())
        loss.backward();

        // Update the model parameters using the gradients.
        params.forEach((p) => p.data += -0.1 * p.grads.get(loss.uniqId));

        // Track our loss for this round.
        losses.push(loss.data);
    })

    const avgLossFirstThird = average(losses.slice(0, losses.length / 3))
    const avgLossMiddleThird = average(losses.slice(losses.length / 3, losses.length / 3 * 2))
    const avgLossLastThird = average(losses.slice(losses.length / 3 * 2, losses.length))

    logger.log('avg loss first third:', avgLossFirstThird)
    logger.log('avg loss middle third:', avgLossMiddleThird)
    logger.log('avg loss last third:', avgLossLastThird)

    // We should see our loss decreasing over time, 
    // because loss is the squared difference between the predicted and ground truth values.
    // In simpler terms -- high loss means the model is very wrong, and low loss means the model is very right.
    expect(avgLossFirstThird).toBeGreaterThan(avgLossLastThird)
});



test('trainer can learn XOR', () => {

    const logger = createSimpleLogger('trainer can learn XOR', false);
    // Our target function is a simple XOR function.
    const targetFunction = (x: number, z: number) => +(x != z);

    // Create the model.
    // Create a simple neural network with two inputs, one hidden layer with two nodes, and one output node.
    const mlp = new MLP(2, [2, 2, 1]);

    // All the params of the whole network.
    const params = mlp.parameters;


    const samples: { input: [Value, Value], groundTruthOutput: [Value] }[] = _.range(0, 1000).map((idx) => {
        const [x, z] = randomArray(2, 0, 1);
        return {
            input: [new Value(x), new Value(z)],
            groundTruthOutput: [new Value(targetFunction(x, z))]
        }
    })

    const trainerResults = trainer<[Value, Value], [Value]>({
        getParameters: () => mlp.parameters,
        samples,
        // TODO augment multi-layer perceptron type system to properly check array types.
        predict: (input) => mlp.run([input[0], input[1]]) as [Value],
        computeLoss: (output, groundTruthOutput) => {
            // Add the outputs of the hidden layer.
            const summedOutput = output.reduce((acc, o) => acc.add(o))

            // Loss is squared difference between the summed output of the last hidden layer,
            // and the ground truth output.
            return summedOutput.sub(groundTruthOutput[0]).pow(2)
        },
        learningRate: 0.05
    })

    const losses = trainerResults.map((r) => r.lossData)

    const avgLossFirstThird = average(losses.slice(0, losses.length / 3))
    const avgLossMiddleThird = average(losses.slice(losses.length / 3, losses.length / 3 * 2))
    const avgLossLastThird = average(losses.slice(losses.length / 3 * 2, losses.length))

    logger.log('avg loss first third:', avgLossFirstThird)
    logger.log('avg loss middle third:', avgLossMiddleThird)
    logger.log('avg loss last third:', avgLossLastThird)

    // We should see our loss decreasing over time, 
    // because loss is the squared difference between the predicted and ground truth values.
    // In simpler terms -- high loss means the model is very wrong, and low loss means the model is very right.
    expect(avgLossFirstThird).toBeGreaterThan(avgLossLastThird)
});