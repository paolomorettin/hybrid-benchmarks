
import torch
from pysmt.shortcuts import *

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


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

    def forward(self, x):
        return self.network(x)


    def to_smt(self, threshold=0.0):

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

        for l in range(int(len(self.network)/2)): # for each layer
            nxt = []
            # for each neuron in layer
            for i in range(self.network[l*2].weight.size()[0]):
                # compute coefficients
                summands = []                
                for j in range(len(prv)): # for each input to the l-th layer
                    w = self.network[l*2].weight[i,j]; assert(w <= 1)
                    if abs(w) > threshold:
                        summands.append(Times(Real(float(w)),
                                             prv[j]))

                if len(summands) > 0:
                    # init variables
                    aux = Symbol(f'a_{l*2}_{i}', REAL)
                    out = Symbol(f'h_{l*2}_{i}', REAL)
                    nxt.append(out)

                    b = self.network[l*2].bias[i]#; assert(b <= 1)
                    lincomb = Plus(*summands, Real(float(b)))

                    formula.append(Equals(aux, lincomb))
                    formula.append(Ite(LE(Real(0), aux),
                                       Equals(out, aux),
                                       Equals(out, Real(0))))
                    formula.append(Not(And(Equals(out, aux),
                                        Equals(out, Real(0)))))
                    formula.append(LE(Real(0), out))
                    formula.append(LE(aux, out))

            prv = nxt

        formula = And(*formula)
        self.output_vars = []        
        for i, hvar in enumerate(prv):
            outvar = Symbol(f'y_{i}', REAL)
            self.output_vars.append(outvar)
            formula = formula.substitute({hvar : outvar})

        return formula


if __name__ == '__main__':

    from sys import argv

    if len(argv) > 1:
        dimensions = [int(x) for x in argv[1:]]

    nn = ReluNet(dimensions, 666)

    print('\n'.join(serialize(clause) for clause in nn.to_smt().args()))
