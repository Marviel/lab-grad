import _ from 'lodash'

interface HasHash<K> {
    hash: () => K
}

function isHasHash<K>(x: any): x is HasHash<K> {
    return _.isFunction(x.hash);
}

/**
 * A map which has a default value for all keys.
 * 
 * If a key is requested that does not exist, it is created with the default value.
 * 
 * Keys can be either a valid hashable item, such as a string, or number.
 * 
 * It is also possible to `get`, `set`, and `getSet` using a `HasHash` object -- which should produce a unique hash.
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
    get(key: K | HasHash<K>): V {
        if (isHasHash(key)) {
            key = key.hash();
        }

        if (!super.has(key)) {
            const v = this._defaultVal();
            super.set(key, this._defaultVal());
            return v;
        }
        // Typescript doesn't know it, but the above check proves that the key exists.
        // @ts-ignore
        return super.get(key);
    }
    set(key: K | HasHash<K>, value: V) {
        if (isHasHash(key)) {
            key = key.hash();
        }

        return super.set(key, value);
    }

    getSet(key: K | HasHash<K>, getNewValue: (v: V) => V) {
        const v = getNewValue(this.get(key));
        this.set(key, v);
    }
}

export class Value implements HasHash<string> {
    uniqId: string = _.uniqueId(`${this.op}_`)
    data: number

    // This is the gradient of the node with respect to a given output.
    // All nodes 
    grads: MapWithDefault<string, number> = new MapWithDefault(0)

    // This is the function that will be called when the backward pass is triggered.
    // The target parameter is the node we're computing gradients for.
    // It should be initialized by the operation that created this node.
    // For leaf nodes, it should be a no-op.
    private _backward: (target: Value) => void = () => { }

    constructor(
        input: Value | number,
        readonly op: string | undefined = undefined,
        readonly children: Value[] = []
    ) {
        this.data = input instanceof Value ? input.data : input
    }

    // This is used to uniquely identify the node.
    hash() {
        return this.uniqId;
    }

    /**
     * Computes all the gradients of the graph with respect to *this* node.
     */
    backward() {
        // Build a topological ordering of the graph.
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

        // The gradient of this node with respect to itself is 1.
        this.grads.set(this.uniqId, 1);

        // Progress backwards through the graph from this node, applying the chain rule.
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
        out._backward = (target: Value) => {
            this.grads.getSet(target, (curVal) => curVal + out.grads.get(target));
            vOther.grads.getSet(target, (curVal) => curVal + out.grads.get(target));
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
        out._backward = (target: Value) => {
            // self.grad += other.data * out.grad
            this.grads.getSet(target, (curVal) => curVal + vOther.data * out.grads.get(target));
            // other.grad += self.data * out.grad
            vOther.grads.getSet(target, (curVal) => curVal + this.data * out.grads.get(target));
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

        out._backward = (target: Value) => {
            // self.grad += (other * self.data**(other-1)) * out.grad
            this.grads.getSet(target, (cur) => cur + (other * (this.data ** (other - 1))) * out.grads.get(target));
        }

        return out
    }

    /**
     * Computes relu of this.data, and returns a Value object representing the result.
     * @returns The resultant Value object.
     */
    public relu(): Value {
        const out = new Value(this.data < 0 ? 0 : this.data, 'ReLU', [this])

        out._backward = (target: Value) => {
            // self.grad += (out.data > 0) * out.grad
            this.grads.getSet(target, (cur) => cur + (this.data > 0 ? out.grads.get(target) : 0));
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

    /**
     * Computes `this.data / other.data`, and returns a Value object representing the result.
     * @param other 
     * @returns 
     */
    public div(other: Value | number): Value {
        const vOther = other instanceof Value ? other : new Value(other)
        return this.mult(vOther.pow(-1));
    }

    /**
     * Computes `other.data / this.data`, and returns a Value object representing the result.
     * @param other The numerator of the division. Will be converted to a Value object if it is not already one.
     * @returns 
     */
    public rev_div(other: Value | number): Value {
        const vOther = other instanceof Value ? other : new Value(other)
        return vOther.mult(this.pow(-1));
    }

    /**
     * Computes `-this.data`, and returns a Value object representing the result.
     * @returns 
     */
    public neg(): Value {
        return this.mult(-1);
    }

    public tanh(): Value {
        const x = this.data
        // t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        const t = (Math.exp(2 * x) - 1) / (Math.exp(2 * x) + 1)
        const out = new Value(t, 'tanh', [this])

        out._backward = (target: Value) => {
            // self.grad += (1 - t**2) * out.grad
            this.grads.getSet(target, (cur) => cur + (1 - t ** 2) * out.grads.get(target));
        }

        return out;
    }

    public exp(): Value {
        const x = this.data;
        // out = Value(math.exp(x), (self, ), 'exp')
        const out = new Value(Math.exp(x), 'exp', [this]);

        // def _backward():
        //   self.grad += out.data * out.grad # NOTE: in the video I incorrectly used = instead of +=. Fixed here.
        out._backward = (target: Value) => {
            this.grads.getSet(target, (cur) => cur + out.data * out.grads.get(target));
        }

        return out;
    }

    /**
     * Gets the first gradient found on this node.
     * 
     * Useful when only calculating gradients for a single node.
     * @returns The first gradient found.
     */
    public get grad(): number | undefined {
        const v = this.grads.entries().next().value;
        return v ? v[1] : undefined;
    }

    // TODO
    // public exp(): Value {
    //     this.data = Math.exp(this.data)

    //     // Backward pass, respective to a particular node.
    //     this._backward = (target: Value) => {
    //         const id = target.uniqId;
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

/**
 * Create a new value from the given data.
 * 
 * If no value is provided, will initialize the value to a random number between -1 and 1.
 * @param input 
 * @returns 
 */
export function newVal(input?: Value | number | undefined): Value {
    return new Value(input !== undefined ? input : _.random(-1, 1, true))
}
