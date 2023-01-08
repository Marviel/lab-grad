import _ from 'lodash'

/**
 * A map which has a default value for all keys, which will be created upon request.
 */
class MapWithDefault<K, V> extends Map<K, V>{
    constructor(readonly defaultVal: V | (() => V), entries?: readonly (readonly [K, V])[]) {
        super(entries);
    }

    _defaultVal(): V {
        return _.isFunction(this.defaultVal) ? this.defaultVal() : this.defaultVal;
    }

    // This function is guaranteed to always return a value, even if the key doesn't exist.
    // The key is also created. if it does not exist.
    get(key: K): V {
        if (!super.has(key)) {
            const v = this._defaultVal();
            super.set(key, this._defaultVal());
            return v;
        }
        // Typescript doesn't know it, but the above check proves that the key exists.
        // @ts-ignore
        return super.get(key);
    }
    set(key: K, value: V) {
        return super.set(key, value);
    }

    getSet(key: K, getNewValue: (v: V) => V) {
        const v = getNewValue(this.get(key));
        this.set(key, v);
    }
}

export class Value {
    uniqId: string = _.uniqueId(`${this.op}_`)
    data: number

    // This is the gradient of the node with respect to a given output.
    // All nodes 
    grads: MapWithDefault<string, number> = new MapWithDefault(0)

    // This is the function that will be called when the backward pass is triggered.
    // The wrt parameter is the node with respect to which the gradient is being computed.
    // It should be initialized by the operation that created this node.
    // For leaf nodes, it should be a no-op.
    private _backward: (wrt: Value) => void = () => { }

    constructor(
        input: Value | number,
        readonly op: string | undefined = undefined,
        readonly children: Value[] = []
    ) {
        this.data = input instanceof Value ? input.data : input
    }

    // def backward(self):

    //     # topological order all of the children in the graph
    //     topo = []
    //     visited = set()
    //     def build_topo(v):
    //         if v not in visited:
    //             visited.add(v)
    //             for child in v._prev:
    //                 build_topo(child)
    //             topo.append(v)
    //     build_topo(self)

    //     # go one variable at a time and apply the chain rule to get its gradient
    //     self.grad = 1
    //     for v in reversed(topo):
    //         v._backward()



    /**
     * Computes all the gradients of the graph with respect to *this* node.
     */
    backward() {
        const topo: Value[] = []
        const visited = new Set()
        const buildTopo = (v: Value) => {
            if (!visited.has(v)) {
                visited.add(v);
                v.children.forEach(buildTopo);
                topo.push(v);
            }
        }
        buildTopo(this);

        this.grads.set(this.uniqId, 1);

        topo.reverse().forEach((v) => v._backward(this));
    }

    /**
     * Computes `this.data + other.data`, and returns a Value object representing the result.
     * @param other The value to add. Will be converted to a Value if it is not already one.
     * @returns The resultant Value object.
     */
    public add(other: Value | number): Value {
        const vOther = other instanceof Value ? other : new Value(other)
        const out = new Value(this.data + vOther.data, '+', [this, vOther])

        // Backward pass, respective to a particular node.
        out._backward = (wrt: Value) => {
            const id = wrt.uniqId;
            this.grads.getSet(id, (curVal) => curVal + out.grads.get(id));
            vOther.grads.getSet(id, (curVal) => curVal + out.grads.get(id));
        }

        return out;
    }



    /**
     * Computes this.data * other.data, and returns a Value object representing the result.
     * @param other The value to multiply. Will be converted to a Value object if it is not already one.
     * @returns The resultant Value object.
     */
    public mult(other: Value | number): Value {
        const vOther = other instanceof Value ? other : new Value(other)
        const out = new Value(this.data * vOther.data, '*', [this, vOther])

        // Backward pass, respective to a particular node.
        out._backward = (wrt: Value) => {
            const id = wrt.uniqId;
            // self.grad += other.data * out.grad
            this.grads.getSet(id, (curVal) => curVal + vOther.data * out.grads.get(id));
            // other.grad += self.data * out.grad
            vOther.grads.getSet(id, (curVal) => curVal + this.data * out.grads.get(id));
        }

        return out;
    }

    /**
     * 
     * NOTE: This only supports numbers for now. 
     * This is because the gradient of a power function with non-constant power is more difficult to compute.
     * 
     * @param other 
     * @returns 
     */
    public pow(other: number): Value {
        const vOther = new Value(other)
        const out = new Value(this.data ** vOther.data, '^', [this, vOther])

        out._backward = (wrt: Value) => {
            const id = wrt.uniqId;
            // self.grad += (other * self.data**(other-1)) * out.grad
            this.grads.getSet(id, (cur) => cur + (other * (this.data ** (other - 1))) * out.grads.get(id));
        }

        return out
    }

    /**
     * Computes relu of this.data, and returns a Value object representing the result.
     * @returns The resultant Value object.
     */
    public relu(): Value {
        const out = new Value(this.data < 0 ? 0 : this.data, 'ReLU', [this])

        out._backward = (wrt: Value) => {
            const id = wrt.uniqId;
            // self.grad += (out.data > 0) * out.grad
            this.grads.getSet(id, (cur) => cur + (this.data > 0 ? out.grads.get(id) : 0));
        }

        return out;
    }

    /**
     * Computes `this.data - other.data`, and returns a Value object representing the result.
     * @param other The value to subtract. Will be converted to a Value object if it is not already one.
     * @returns The resultant Value object.
     */
    public sub(other: Value | number): Value {
        const vOther = other instanceof Value ? other : new Value(other)
        return this.add(vOther.neg());
    }

    public div(other: Value | number): Value {
        const vOther = other instanceof Value ? other : new Value(other)
        return this.mult(vOther.pow(-1));
    }

    public rev_div(other: Value | number): Value {
        const vOther = other instanceof Value ? other : new Value(other)
        return vOther.mult(this.pow(-1));
    }

    public neg(): Value {
        return this.mult(-1);
    }

    // TODO
    // public exp(): Value {
    //     this.data = Math.exp(this.data)

    //     // Backward pass, respective to a particular node.
    //     this._backward = (wrt: Value) => {
    //         const id = wrt.uniqId;
    //         // self.grad += (other * self.data**(other-1)) * out.grad
    //         this.grads.getSet(id, (cur) => cur + (other * (this.data ** (other - 1))) * out.grads.get(id));
    //     }

    //     return new Value(this.data, 'exp', [this])
    // }

    // TODO
    // public log(): Value {
    //     this.data = Math.log(this.data)
    //     return new Value(this.data, 'log', [this])
    // }

    // TODO
    // public sin(): Value {
    //     this.data = Math.sin(this.data)
    //     return new Value(this.data, 'sin', [this])
    // }

    // TODO
    // public cos(): Value {
    //     this.data = Math.cos(this.data)
    //     return new Value(this.data, 'cos', [this])
    // }

    // TODO
    // public tan(): Value {
    //     this.data = Math.tan(this.data)
    //     return new Value(this.data, 'tan', [this])
    // }
}

export function newVal(input: Value | number): Value {
    return new Value(input)
}
