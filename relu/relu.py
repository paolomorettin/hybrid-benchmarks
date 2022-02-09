
import torch
from pysmt.shortcuts import *

class ModulePlus(torch.nn.Module):

    def __init__(self, seed):
        torch.manual_seed(seed)
        super(ModulePlus, self).__init__()

    def trainable_params(self):
        return sum(dict((p.data_ptr(), p.numel())
                        for p in self.parameters()).values())

class ReluNet(ModulePlus):
    def __init__(self, dimensions, seed):
        '''
        A feed-forward network with ReLU activations with:

        - (len(dimensions)-1) layers
        - the ith-layer has dimensions[i] variables

        '''
        super(ReluNet, self).__init__(seed)
        assert(len(dimensions) > 1), "input/output dimensions unspecified"
        stack = []
        self.dimensions = dimensions

        for i in range(len(dimensions) - 1):
            stack.extend([torch.nn.Linear(dimensions[i], dimensions[i+1]),
                          torch.nn.ReLU()])        
            
        self.network = torch.nn.Sequential(*stack)
        self.to_smt()

    def forward(self, x):
        return self.network(x)


    def to_smt(self):

        assert(len(self.network) % 2 == 0)
        assert(all(isinstance(l, torch.nn.Linear)
                   for i, l in enumerate(self.network)
                   if i % 2 == 0))
        assert(all(isinstance(l, torch.nn.ReLU)
                   for i, l in enumerate(self.network)
                   if i % 2 == 1))

        formula = []
        
        # input vars
        prv = [Symbol(f'x_{i}', REAL)
               for i in range(self.network[0].weight.size()[1])]

        self.input_vars = list(prv)

        for l in range(int(len(self.network)/2)):
            # linear combinations
            nxt = []
            for i in range(self.network[l*2].weight.size()[0]):
                aux = Symbol(f'a_{l*2}_{i}', REAL)
                out = Symbol(f'h_{l*2}_{i}', REAL)
                b = Real(float(self.network[l*2].bias[i]))
                wx = Plus(*[Times(Real(float(self.network[l*2].weight[i,j])),
                                  prv[j]) for j in range(len(prv))])
                nxt.append(out)
                formula.append(Equals(aux, Plus(b, wx)))
                formula.append(Ite(LE(Real(0), aux),
                                   Equals(out, aux),
                                   Equals(out, Real(0))))
                formula.append(LE(Real(0), out))
                formula.append(LE(aux, out))

            prv = nxt

        for clause in formula:
            print('   ', serialize(clause))

        formula = And(*formula)
        self.output_vars = []        
        for i, hvar in enumerate(prv):
            outvar = Symbol(f'y_{i}', REAL)
            self.output_vars.append(outvar)
            formula = formula.substitute({hvar : outvar})

        self.formula = formula


if __name__ == '__main__':

    from sys import argv

    if len(argv) > 1:
        dimensions = [int(x) for x in argv[1:]]

    nn = ReluNet(dimensions, 666)

    print('\n'.join(serialize(clause) for clause in nn.to_smt().args()))
