import torch


def test1():
    print("----------------")
    print("test1")
    print("----------------")
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    print('ypt.data.item() =', ypt.data.item())
    print('xpt.grad.item() =', xpt.grad.item())


def test2():
    print("----------------")
    print("test2")
    print("----------------")
    a = torch.Tensor([-4.0]).double()
    print('a', a.data.item())
    b = torch.Tensor([2.0]).double()
    print('b', b.data.item())
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    print('c', c.data.item())
    d = a * b + b**3
    print('d', d.data.item())
    c = c + c + 1
    print('c1', c.data.item())
    c = c + 1 + c + (-a)
    print('c2', c.data.item())
    d = d + d * 2 + (b + a).relu()
    print('d1', d.data.item())
    d = d + 3 * d + (b - a).relu()
    print('d2', d.data.item())
    e = c - d
    print('e', e.data.item())
    f = e**2
    print('f', f.data.item())
    g = f / 2.0
    print('g', g.data.item())
    g = g + 10.0 / f
    print('g1', g.data.item())
    g.backward()

    # forward pass went well
    print('g.data.item() = ', g.data.item())
    # backward pass went well
    print('a.grad.item() = ', a.grad.item())
    print('b.grad.item() = ', b.grad.item())


test1()
test2()