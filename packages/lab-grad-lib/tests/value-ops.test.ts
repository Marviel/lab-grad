import { newVal, Value } from '../src/Value'

test('adds 1 + 2 to equal 3', () => {
    expect(newVal(1).add(2).data).toBe(3);
});

test('add op is correct', () => {
    expect(newVal(1).add(2).op).toBe('+');
});

test('subtracts 3 - 2 to equal 1', () => {
    expect(newVal(3).sub(2).data).toBe(1);
});

test('sub op is correct', () => {
    // This is still '+' because it reduces to a negation of the second term under the hood.
    expect(newVal(3).sub(2).op).toBe('+');
});

test('multiplies 3 * 2 to equal 6', () => {
    expect(newVal(3).mult(2).data).toBe(6);
});

test('mult op is correct', () => {
    expect(newVal(3).mult(2).op).toBe('*');
})

test('divides 6 / 2 to equal 3', () => {
    expect(newVal(6).div(2).data).toBe(3);
});

test('negates 3 to equal -3', () => {
    expect(newVal(3).neg().data).toBe(-3);
});

test('negates -3 to equal 3', () => {
    expect(newVal(-3).neg().data).toBe(3);
});

test('neg op is correct', () => {
    // Neg reduces to a multiplication by -1 under the hood.
    expect(newVal(3).neg().op).toBe('*');
})

test('exponentiates 2 ** 3 to equal 8', () => {
    expect(newVal(2).pow(3).data).toBe(8);
});

test('pow op is correct', () => {
    expect(newVal(2).pow(3).op).toBe('^');
});

test('relu of 3 to equal 3', () => {
    expect(newVal(3).relu().data).toBe(3);
});

test('relu of -3 to equal 0', () => {
    expect(newVal(-3).relu().data).toBe(0);
});

test('relu op is correct', () => {
    expect(newVal(3).relu().op).toBe('ReLU');
});