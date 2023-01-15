import _ from "lodash";

export const average = (array: number[]) => array.reduce((a, b) => a + b) / array.length;

export function randomArray(length: number, low: number, high: number) {
    return _.range(0, length).map(() => _.random(low, high));
}