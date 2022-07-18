# from utils import *

# x_cstr_string = ', '. join([f'x{get_sub(str(i))}' for i in range(10)]) + ' ' + u'\u2265' + ' ' + '0'
# print(x_cstr_string)

# debug solver
# optimal 
# c = [2,1]
# b = [4,3,5,1]
# N = [[2,1],
#      [2,3],
#      [4,2],
#      [1,5]]

# infeasible
# c = [1,3]
# N = [[-1,-1],
#      [-1,1],
#      [1,2],
#      ]
# b = [-3,-1,2]

# unbounded
# c = [1,-1]
# N = [[-2,3],
#      [0,4],
#      ]
# b = [5,7]

# # dual optimal
# c = [-1, -3, -1]
# N = [[2, -5, 1],
#      [2, -1, 2]]

# b = [-5,4]

# # two phase optimal

# c = [2, -6, 0]
# N = [[-1, -1, -1],
#      [2, -1, 1]]

# b = [-2,1]

# Unbounded
c = [1, 3]
N = [[-1,-1],
    [-1,1],
    [-1,2]]
b = [-3,-1,2]