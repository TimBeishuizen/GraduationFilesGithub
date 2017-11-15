#inv_loc = np.unique(np.all([['A'], ['N']], [['C'], ['T']] not in [['A'], ['G'], ['T'], ['C']])[0])
#print(inv_loc)

import numpy as np

# inv_loc = np.unique([['A'],['N'],['N'],['G']]) not in [['A'], ['G'], ['T'], ['C']]



# print(list(x not in [['A'], ['G'], ['T'], ['C']] for x in [['A'],['N'],['N'],['G']]))

x = ([['A','N'],['T','C']])
y = [['A'], ['G'], ['T'], ['C']]

print(np.isin(x, y))

print(np.array(np.where(x == ['A','G'])))