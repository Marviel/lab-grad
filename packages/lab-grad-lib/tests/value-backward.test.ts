import { newVal } from "../src/Value";

test('1 + 2 backward works correctly', () => {
    const v = newVal(1).add(2);

    v.backward();

    // This node should have a gradient of 1.
    expect(v.grads.get(v.uniqId)).toBe(1);

    // The left child should have a gradient of 1.
    expect(v.children[0].grads.get(v.uniqId)).toBe(1);

    // The right child should have a gradient of 1.
    expect(v.children[1].grads.get(v.uniqId)).toBe(1);
});

test('this + this backward works correctly', () => {
    const a = newVal(1);
    const b = a.add(a);

    b.backward();

    // This node should have a gradient of 1 wrt B.
    expect(a.grads.get(b.uniqId)).toBe(2);

    // This node should have a gradient of 2.
    expect(b.grads.get(b.uniqId)).toBe(1);
})


test('pytorch basic backward works correctly', () => {
    // x = torch.Tensor([-4.0]).double()
    // x.requires_grad = True
    // z = 2 * x + 2 + x
    // q = z.relu() + z * x
    // h = (z * z).relu()
    // y = h + q + q * x
    // y.backward()
    // xpt, ypt = x, y

    const x = newVal(-4.0);
    const z = x.mult(2).add(2).add(x);
    const q = z.relu().add(z.mult(x));
    const h = z.mult(z).relu();
    const y = h.add(q).add(q.mult(x));
    y.backward();

    expect(x.grads.get(y.uniqId)).toBeCloseTo(46.0);
    expect(y.data).toBeCloseTo(-20);
});

test('failing case works correctly', () => {
    const a = newVal(-4);
    const b = newVal(2);
    const d1 = newVal(0);

    const d2 = d1.add(d1.mult(3)).add(b.sub(a).relu());
    expect(d2.data).toBe(6);
});

test('pytorch advanced backward works correctly', () => {
    // a = torch.Tensor([-4.0]).double()
    // b = torch.Tensor([2.0]).double()
    // a.requires_grad = True
    // b.requires_grad = True
    // c = a + b
    // d = a * b + b**3
    // c = c + c + 1
    // c = c + 1 + c + (-a)
    // d = d + d * 2 + (b + a).relu()
    // d = d + 3 * d + (b - a).relu()
    // e = c - d
    // f = e**2
    // g = f / 2.0
    // g = g + 10.0 / f
    // g.backward()
    const a = newVal(-4.0);
    const b = newVal(2.0);
    const c = a.add(b);
    const d = a.mult(b).add(b.pow(3));
    const c1 = c.add(c).add(1);
    const c2 = c1.add(1).add(c1).add(a.neg());
    const d1 = d.add(d.mult(2)).add(b.add(a).relu());
    const d2 = d1.add(d1.mult(3)).add(b.sub(a).relu());
    const e = c2.sub(d2);
    const f = e.pow(2);
    const g = f.div(2.0);
    const g2 = g.add(f.rev_div(10.0));
    g2.backward();

    expect(g2.data).toBeCloseTo(24.70408163265306);
    expect(a.grads.get(g2.uniqId)).toBeCloseTo(138.83381924198252);
    expect(b.grads.get(g2.uniqId)).toBeCloseTo(645.5772594752186);
});