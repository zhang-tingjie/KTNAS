from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'Identity',
    'AVG',
    'MAX',
    'DW3',
    'DW5',
    'DW7',
    'DC3',
    'DC5',
    'DC7'
]

EMT_C10 = Genotype(normal=[('DW5', 0), ('DC3', 0), ('DW3', 0), ('DC7', 0), ('DC7', 0), ('DC3', 1), ('DW5', 0), ('DC3', 1), ('DW5', 2), ('DW5', 2)], normal_concat=[3, 4, 5, 6], reduce=[('DW5', 0), ('DW5', 1), ('DC3', 2), ('DC7', 1), ('DW3', 0), ('DC7', 1), ('DW5', 0), ('DW5', 1), ('DW3', 0), ('DW5', 3)], reduce_concat=[4, 5, 6])
EMT_C100 = Genotype(normal=[('DW5', 0), ('DC3', 0), ('DW3', 0), ('DC7', 2), ('DW5', 2), ('DW5', 2), ('DC3', 1), ('DC7', 0), ('DW5', 0), ('DC7', 2)], normal_concat=[3, 4, 5, 6], reduce=[('DW5', 0), ('DW5', 1), ('DC7', 2), ('DW5', 2), ('DW3', 0), ('DC7', 1), ('DW5', 0), ('DW5', 1), ('DW3', 0), ('DW7', 3)], reduce_concat=[4, 5, 6])

KT_C10 = Genotype(normal=[('Identity', 1), ('AVG', 1), ('MAX', 1), ('DW3', 1), ('DW3', 1), ('DC7', 2), ('DC5', 0), ('MAX', 4), ('DW3', 0), ('DW7', 4)], normal_concat=[3, 5, 6], reduce=[('MAX', 1), ('DC7', 1), ('Identity', 0), ('DC5', 1), ('DW5', 0), ('AVG', 0), ('AVG', 4), ('MAX', 3), ('DC3', 3), ('MAX', 0)], reduce_concat=[2, 5, 6])
KT_C100 = Genotype(normal=[('DC5', 0), ('DC7', 0), ('DW5', 1), ('DW7', 0), ('AVG', 1), ('MAX', 0), ('DW7', 1), ('Identity', 0), ('DW5', 1), ('AVG', 4)], normal_concat=[2, 3, 5, 6], reduce=[('DC7', 1), ('DW3', 1), ('DW3', 0), ('DW3', 1), ('DC5', 0), ('AVG', 2), ('DC3', 0), ('DC3', 4), ('DW3', 2), ('DC3', 1)], reduce_concat=[3, 5, 6])
