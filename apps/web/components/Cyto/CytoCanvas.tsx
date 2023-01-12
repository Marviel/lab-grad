import dynamic from 'next/dynamic';
import React from 'react';
const CytoscapeComponent = dynamic(() => import('react-cytoscapejs'), { ssr: false });
import cytoscape from 'cytoscape';
import klay from 'cytoscape-klay';
import { CytoscapeOptions } from 'cytoscape';
import { newVal, Value } from '@lab-grad/lib';
import colormap from 'colormap';

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
            label: v.data.toString(),
            backgroundColor: getColor(v.data, 10),
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

export function CytoCanvas() {
    //@ts-ignore
    cytoscape.use(klay);
    const elements = ValueToCytoElements(newVal(1).add(2).sub(3));

    return <CytoscapeComponent
        cy={(cy) => { cy.fit }}
        elements={elements}
        style={{ width: '100%', height: '100%', backgroundColor: '#039956' }}
        layout={{ name: 'klay' }}
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
                    'color': 'gray',
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
    />;
}