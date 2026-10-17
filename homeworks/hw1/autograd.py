"""Задание 2. Скалярный autograd: сложение, умножение и ReLU."""


class Node:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Node(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # чтобы работали выражения вида 2 + a и 3 * a
    __radd__ = __add__
    __rmul__ = __mul__

    def relu(self):
        out = Node(self.data if self.data > 0 else 0, (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # топологическая сортировка: градиент узла считается только после всех его потребителей
        order, visited = [], set()

        def visit(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    visit(child)
                order.append(node)

        visit(self)
        self.grad = 1
        for node in reversed(order):
            node._backward()


if __name__ == "__main__":
    a = Node(2)
    b = Node(-3)
    c = Node(10)
    d = a + b * c
    e = d.relu()
    e.backward()
    print(a, b, c, d, e)
