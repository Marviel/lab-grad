import dynamic from 'next/dynamic';
import React, { useEffect, useMemo, useState } from 'react';
const CytoscapeComponent = dynamic(() => import('react-cytoscapejs'), { ssr: false });
import cytoscape from 'cytoscape';
import klay from 'cytoscape-klay';
import { CytoscapeOptions } from 'cytoscape';
import { MLP, newVal, randomArray, Trainer, trainer, TrainerSampleResult, Value } from '@lab-grad/lib';
import colormap from 'colormap';
import _ from 'lodash';
import { Button } from 'ui';

const colors = colormap({
    colormap: 'jet',
    nshades: 10,
    format: 'hex',
    alpha: 1
})

const getColor = (v: number, maxV: number) => {
    const index = Math.min(Math.floor((v / maxV) * colors.length), colors.length - 1);
    return colors[index];
}

// Get the type of the array elements in typescript
type Unarray<T> = T extends (infer U)[] ? U : never;

function ValueToCytoNode(v: Value): cytoscape.ElementDefinition {
    return {
        data: {
            id: v.uniqId,
            label: v.data.toFixed(2),
            backgroundColor: getColor(v.data, 2),
            shape: 'ellipse'
        },
        // position: { x: 0, y: 0 }
    }
}

function ValueToCytoElements(v: Value): cytoscape.ElementDefinition[] {
    // Recursively fetch children.
    const childResults = v.children.map((childV) => ValueToCytoElements(childV));

    const opId = v.uniqId + '_op';


    return [
        // Add self
        ValueToCytoNode(v),
        // Add op node.
        ...(v.op ?
            [
                {
                    data: {
                        source: v.uniqId,
                        target: opId,
                        // label: 'YOOO',
                        color: 'gray'
                    }
                },
                {
                    data: {
                        id: opId,
                        label: v.op,
                        backgroundColor: '#CCCCCC',
                        color: 'black',
                        shape: 'diamond'
                    }
                },
                // Add edge to child results.
                ...v.children.map((childResult) => ({
                    data: {
                        source: opId,
                        target: childResult.uniqId,
                        // label: 'YOOO',
                    }
                })),
                // Add child results.
                ...childResults.flat()
            ]
            :
            []
        ),
    ]
}

function styleForSelector(elType: string, name: string) {
    return {
        selector: `${elType}[${name}]`,
        style: {
            [name]: `data(${name})`,
        }
    }
}

function getAllSelectors(elType: string, names: string[]) {
    return names.map((name) => styleForSelector(elType, name));
}


function getTrainer(sampleCompleteCb: (sample: TrainerSampleResult & { loss: Value }) => void) {
    // Our target function is a simple XOR function.
    const targetFunction = (x: number, z: number) => +(x != z);

    // Create the model.
    // Create a simple neural network with two inputs, one hidden layer with three nodes, and one output node.
    const mlp = new MLP(2, [2, 1]);

    const samples: { input: [Value, Value], groundTruthOutput: [Value] }[] = _.range(0, 10000).map((idx) => {
        const [x, z] = randomArray(2, 0, 1);
        return {
            input: [new Value(x), new Value(z)],
            groundTruthOutput: [new Value(targetFunction(x, z))]
        }
    })

    return new Trainer({
        getParameters: () => mlp.parameters,
        samples,
        // TODO augment multi-layer perceptron type system to properly check array types.
        predict: (input) => mlp.run([input[0], input[1]]) as [Value],
        computeLoss: (output, groundTruthOutput) => {
            // Loss is squared difference between the summed output of the last hidden layer,
            // and the ground truth output.
            return output[0].sub(groundTruthOutput[0]).pow(2)
        },
        learningRate: 0.05,
        stepTime: 1000,
        sampleCompleteCallback: sampleCompleteCb,
    })
}


export function CytoCanvas() {
    //@ts-ignore
    cytoscape.use(klay);
    const [lastSample, setLastSample] = useState<(TrainerSampleResult & { loss: Value }) | undefined>(undefined);

    const trainer = useMemo(() => getTrainer((sample) => {
        setLastSample(sample);
    }), [])

    const rootV = useMemo(() => {
        return lastSample ? lastSample.loss : undefined;
    }, [lastSample])

    // useEffect(() => {
    //     setInterval(() => setRootV(rootV.add(Math.random())), 5000);
    // }, [])

    const cytoElements = useMemo(() => rootV ? ValueToCytoElements(rootV) : [], [rootV]);

    function handleCy(cy: cytoscape.Core) {
        const SELECT_THRESHOLD = 100;

        // Refresh Layout if needed
        const refreshLayout = _.debounce(() => {
            //@ts-ignore
            cy.layout({ name: 'klay', klay: { direction: 'LEFT' } }).run();
        }, SELECT_THRESHOLD);

        cy.on('add remove', () => {
            refreshLayout();
        });
    }

    // useEffect(() => {


    //     // console.log(layout);
    //     // if (layout) {
    //     //     klay.run();
    //     // }
    // }, [cy, cytoElements])



    return <div style={{ height: '100vw', width: '100vw' }}>
        <CytoscapeComponent
            cy={(cy) => { handleCy(cy) }}
            elements={cytoElements}
            style={{ width: '100%', height: '50%', backgroundColor: '#039956' }}
            layout={{ name: 'klay' }}
            options={{
                direction: 'UP'
            }}
            // zoomingEnabled={true}
            maxZoom={4}
            zoom={.5}
            stylesheet={[
                {
                    selector: 'node',
                    style: {
                        // label: 'data(label)',
                        width: 10,
                        height: 10,
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-family': 'Lato, Verdana, Geneva, sans-serif',
                        'font-size': '2vw',
                        'color': 'black',
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        width: 1,
                        'curve-style': 'bezier',
                        'font-family': 'Lato, Verdana, Geneva, sans-serif',
                        'font-size': '2vw',
                    }
                },
                {
                    selector: 'node[backgroundColor]',
                    style: {
                        'background-color': 'data(backgroundColor)',
                        'text-outline-color': 'data(backgroundColor)',
                    }
                },
                ...getAllSelectors('edge', ['label', 'color', 'width', 'line-color', 'target-arrow-color', 'target-arrow-shape', 'source-arrow-color', 'source-arrow-shape', 'curve-style']),
                ...getAllSelectors('node', ['shape', 'color', 'label', 'backgroundColor']),
            ]}
        />
        <div>
            <Button onClick={() => trainer.runNextStep()}>Take Step</Button>
            <div>
                {lastSample && <div>
                    <div>Loss: {lastSample.loss.data}</div>
                    <div>Input: {lastSample.sample.input.map((v) => v.data).join(', ')}</div>
                    <div>Ground Truth Output: {lastSample.sample.groundTruthOutput.map((v) => v.data).join(', ')}</div>
                    <div>Output: {lastSample.predictedOutput.map((o) => o).join(', ')}</div>
                </div>}
            </div>
        </div>
    </div>
}