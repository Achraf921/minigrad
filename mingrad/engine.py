import math

class Value:
    def __init__(self,data,_children=(), _op='', label=''): #default empty tuple for children, empty string for op
        self.data=data
        self._prev = set(_children)
        self._op=_op
        self._backward = lambda: None
        self.label=label
        self.grad = 0.0 
    def __repr__(self):
        return f"Value(data={self.data})"
    #for all gradient calculations, we have to use +='s in the scenario where a value gets passed over twice
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # to allow expressions of the kind a+8
        out = Value(self.data+other.data, (self,other), _op='+')
        def _backward(): 
            self.grad +=1.0 * out.grad
            other.grad +=1.0 * out.grad
        out._backward  = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # to allow expressions of the kind a*8
        out = Value(self.data*other.data, (self,other), _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward=_backward
        return out
    def tanh(self):
        out = Value((math.exp(2*self.data)-1)/(math.exp(2*self.data)+1), (self,), _op='tanh')
        def _backward():
            self.grad += (1-(out.data**2)) * out.grad
        out._backward=_backward
        return out

    def __pow__(self, pow): #value**k
        assert isinstance(pow, (int,float)), "only supporting int and float powers"
        out = Value(self.data**pow, (self, ), _op=f"**{pow}")
        def _backward():  
                self.grad += pow*(self.data**(pow-1)) * out.grad
        out._backward=_backward
        return out   

    def __neg__(self): #-self
        return self * -1
    
    def __sub__(self,other):#self - other  
        other = other if isinstance(other,Value) else Value(other) 
        return self + (-other)

    
 
    def __rmul__(self,other): #other*self
        return self*other #to allow 2 * a kind of operations. 
    
    __radd__ = __add__

    def exp(self):
        out = Value(math.exp(self.data), (self,),_op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward=_backward
        return out

    def __truediv__(self, other): #self/other
        return self*(other**-1)

    def backward(self): # only on DAGs
        topo = []
        visited = set()
        def _build_topo(value):
            if value not in visited:
                visited.add(value)
                for child in value._prev:
                    _build_topo(child)
                topo.append(value)
        _build_topo(self)
        self.grad=1.0; #assuming autograd is called on the output node
        topo.reverse() # Post-fix DFS -> Topological Search
        for value in topo:
            value._backward() #.  iterative autograd