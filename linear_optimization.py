import numpy as np
from utils import *
from copy import copy

def get_standard_form(min_max, obj_coeff, constraints_coeff, signs, rhs):
    obj_coeff_arr = np.array(obj_coeff)
    constraints_coeff_arr = np.array(constraints_coeff)
    rhs_arr = np.array(rhs)
    min_max_standard = copy(min_max)
    signs_standard = copy(signs)

    if min_max=='min':
        obj_coeff_arr = -obj_coeff_arr
        min_max_standard = 'max'

    for i, sign in enumerate(signs):
        # larger than
        if sign == u'\u2265': 
            signs_standard[i] = u'\u2264'
            constraints_coeff_arr[i] = -constraints_coeff_arr[i]
            rhs_arr[i] = -rhs_arr[i]
        elif sign == '=':
            # convert "equal" to "smaller or equal" 
            signs_standard[i] = u'\u2264'
            
            # add "larger or equal" (directly convert to the smaller or equal form)
            signs_standard.append(u'\u2264')
            constraints_coeff_arr = np.append(constraints_coeff_arr, np.expand_dims(-constraints_coeff_arr[i], axis=0), axis=0)
            rhs_arr = np.append(rhs_arr, np.expand_dims(-rhs_arr[i], axis=0), axis=0) 

    print(min_max_standard)
    print(obj_coeff_arr)
    print(constraints_coeff_arr)
    print(signs_standard)
    print(rhs_arr)

    return constraints_coeff_arr, rhs_arr, obj_coeff_arr
    

def primal(N_index, B_index, A, b, cN, xB_star, zN_star, N0 = None, B0 = None, verbose=True):
    solution_str = ''

    B = A[:,B_index.astype('uint')-1]
    N = A[:,N_index.astype('uint')-1]

    if N0 is None:
        N0 = N_index.astype('uint')

    assert (xB_star>=0).all()

    while ~(zN_star>=0).all():
        solution_str = print_and_log('Iteration', solution_str)
        solution_str = print_and_log('Step 1: zN* is not >= 0', solution_str)

        j_temp = np.where(zN_star<0)[0][0]
        j = N_index[j_temp]

        solution_str = print_and_log(f'Step 2: j={j} because z{j}*={zN_star[j_temp]}, x{j} is entering', solution_str)

        ej = np.zeros(N_index.size)
        ej[j_temp] = 1
        delta_xB = np.matmul(np.matmul(np.linalg.inv(B), N), ej).squeeze()
        
        solution_str = print_and_log(f'Step 3: delta_xB=\n{delta_xB}', solution_str)

        i_temp = np.argmax(delta_xB/xB_star)
        
        t = (delta_xB/xB_star)[i_temp]**(-1)
        if (delta_xB/xB_star)[i_temp]<=0:
            if verbose:
                solution_str = print_and_log('', solution_str)
                solution_str = print_and_log('Unbounded', solution_str)

            return "Unbounded", None, None, B, N, B_index, N_index, xB_star, zN_star, solution_str

        solution_str = print_and_log(f'Step 4: t={t}', solution_str)

        i = B_index[i_temp]

        solution_str = print_and_log(f'Step 5: leaving x{i} because t max at i={i}', solution_str)

        ei = np.zeros(B_index.size)
        ei[i_temp] = 1
        delta_zN = -np.matmul(np.matmul(np.linalg.inv(B), N).T, ei).squeeze()
        
        solution_str = print_and_log(f'Step 6: delta_zN=\n{delta_zN}', solution_str)

        s = zN_star[j_temp]/delta_zN[j_temp]
        
        solution_str = print_and_log(f'Step 7: t={t}', solution_str)    
        solution_str = print_and_log(f'Step 8: x*{j}={t}', solution_str)

        xB_star = xB_star - t*delta_xB

        solution_str = print_and_log(f'xB_star={xB_star}', solution_str)
        solution_str = print_and_log(f'z*{i}={s}', solution_str)

        zN_star = zN_star - s*delta_zN

        solution_str = print_and_log(f'zN_star={zN_star}', solution_str)


        B_index[i_temp], N_index[j_temp] = j, i
        solution_str = print_and_log(f'Step 9: B_index{B_index}, N_index{N_index}', solution_str)

        B = A[:,B_index.astype('uint8')-1]
        N = A[:,N_index.astype('uint8')-1]

        xB_star[i_temp] = t
        zN_star[j_temp] = s

        solution_str = print_and_log(f'B=\n{B}\nN=\n{N}\nxB_star=\n{xB_star}\nzN_star=\n{zN_star}', solution_str)
        solution_str = print_and_log(f'Simplex Coeff \n{np.matmul(np.linalg.inv(B), N)}\n', solution_str)


    solution = np.zeros_like(N_index)
    mask = np.isin(B_index.astype('uint'), N0)
    index = B_index[mask]
    value = xB_star[mask]


    solution[index.astype('uint')-1] = value

    optimal_value = np.sum(cN * solution)

    if verbose:
        solution_str = print_and_log(f"Solution: {solution}", solution_str)
        solution_str = print_and_log(f"Optimal value: {optimal_value}", solution_str)

    return "Optimal", solution, optimal_value, B, N, B_index, N_index, xB_star, zN_star, solution_str

def dual(N_index, B_index, A, b, cN, xB_star, zN_star, N0 = None, B0 = None, verbose=True):
    solution_str = ''

    B = A[:,B_index.astype('uint')-1]
    N = A[:,N_index.astype('uint')-1]
    if N0 is None:
        N0 = N_index.astype('uint')

    assert (zN_star>=0).all(), "Must input true form of dual"

    while ~(xB_star>=0).all():
        solution_str = print_and_log('Iteration', solution_str)

        solution_str = print_and_log('Step 1: xB* is not >= 0', solution_str)
        
        i_temp = np.where(xB_star<0)[0][0]
        i = B_index[i_temp]

        solution_str = print_and_log(f'Step 2: i={i} because x{i}*={xB_star[i_temp]}, z{i} is entering', solution_str)

        ei = np.zeros(B_index.size)
        ei[i_temp] = 1
        delta_zN = -np.matmul(np.matmul(np.linalg.inv(B), N).T, ei).squeeze()

        solution_str = print_and_log(f'Step 3: delta_zN=\n{delta_zN}', solution_str)

        j_temp = np.argmax(delta_zN/zN_star)
        s = (delta_zN/zN_star)[j_temp]**(-1)
        if (delta_zN/zN_star)[j_temp] <= 0:
            if verbose:
                solution_str = print_and_log('', solution_str)
                solution_str = print_and_log('Primal infeasible', solution_str)
            return "Primal infeasible", None, None, B, N, B_index, N_index, xB_star, zN_star, solution_str
        
        solution_str = print_and_log(f'Step 4: s={s}', solution_str)
        j = N_index[j_temp]

        solution_str = print_and_log(f'Step 5: leaving z{j} because s max at j={j}', solution_str)
        ej = np.zeros(N_index.size)
        ej[j_temp] = 1
        delta_xB = np.matmul(np.matmul(np.linalg.inv(B), N), ej).squeeze()

        solution_str = print_and_log(f'Step 6: delta_xB=\n{delta_xB}', solution_str)

        t = xB_star[i_temp]/delta_xB[i_temp]

        solution_str = print_and_log(f'Step 7: t={t}', solution_str)  
        solution_str = print_and_log(f'Step 8: x*{j}={t}', solution_str)

        xB_star = xB_star - t*delta_xB

        solution_str = print_and_log(f'xB_star={xB_star}', solution_str)
        solution_str = print_and_log(f'z*{i}={s}', solution_str)
        
        zN_star = zN_star - s*delta_zN
        
        solution_str = print_and_log(f'zN_star={zN_star}', solution_str)
        
        B_index[i_temp], N_index[j_temp] = j, i

        solution_str = print_and_log(f'Step 9: B_index{B_index}, N_index{N_index}', solution_str)

        B = A[:,B_index.astype('uint8')-1]
        N = A[:,N_index.astype('uint8')-1]

        xB_star[i_temp] = t
        zN_star[j_temp] = s

        solution_str = print_and_log(f'B=\n{B}\nN=\n{N}\nxB_star=\n{xB_star}\nzN_star=\n{zN_star}', solution_str)        
        solution_str = print_and_log('', solution_str)

    solution = np.zeros_like(N_index)
    mask = np.isin(B_index.astype('uint'), N0)

    index = B_index[mask]
    value = xB_star[mask]

    solution[index.astype('uint')-1] = value

    optimal_value = np.sum(cN * solution)

    if verbose:
        solution_str = print_and_log(f'Solution: {solution}', solution_str)
        solution_str = print_and_log(f'Optimal value: {optimal_value}', solution_str)

    return "Optimal", solution, optimal_value, B, N, B_index, N_index, xB_star, zN_star, solution_str

def two_phase(N_index, B_index, A, b, cN, xB_star, zN_star, N0 = None, B0 = None, verbose=True):
    solution_str = ''

    B = A[:,B_index.astype('uint')-1]
    N = A[:,N_index.astype('uint')-1]

    N0 = N_index.astype('uint')
    B0 = B_index.astype('uint')

    c = np.zeros(N_index.shape[0]+B_index.shape[0])
    c[:N_index.shape[0]] = cN


    cN_modified = np.full_like(cN, fill_value=-1)
    zN_star = -cN_modified
    dual_status, x_max, max, B, N, B_index, N_index, xB_star, zN_star, solution_str_dual = dual(N_index=N_index, B_index=B_index,
                A=A, b=b, cN=cN_modified, xB_star=xB_star, zN_star=zN_star, verbose=False)
    
    solution_str += solution_str_dual

    if dual_status == "Primal infeasible":
        solution_str = print_and_log("", solution_str) 
        solution_str = print_and_log('Infeasible', solution_str)

        return 'Infeasible', None, None, B, N, B_index, N_index, xB_star, zN_star, solution_str
    else:
        cb = c[B_index.astype('uint')-1]
        cn = c[N_index.astype('uint')-1]

        zN_star = np.matmul(np.linalg.inv(B), N).T.dot(cb) - cn
    
        status, solution, optimal_value, B, N, B_index, N_index, xB_star, zN_star, solution_str_primal =  primal(N_index=N_index, B_index=B_index,
                        A=A, b=b, cN=cN, xB_star=xB_star, zN_star=zN_star, N0=N0, B0=B0, verbose=False)
        
        solution_str += solution_str_primal

        if verbose:
            if status == 'Unbounded':
                solution_str = print_and_log('', solution_str)         
                solution_str = print_and_log('Unbounded', solution_str)
            else:
                solution_str = print_and_log(f'Solution: {solution}', solution_str)
                solution_str = print_and_log(f'Optimal value: {optimal_value}', solution_str)
            
        return status, solution, optimal_value, B, N, B_index, N_index, xB_star, zN_star, solution_str


def preprocess(N, b, cN):
    N = np.array(N).astype('float')

    b = np.array(b).astype('float')
    cN = np.array(cN).astype('float')

    N_index = np.arange(1, N.shape[1]+1).astype('float')
    B_index = np.arange(N.shape[1]+1, N.shape[1]+ N.shape[0]+1).astype('float')

    B = np.identity(n=N.shape[0]).astype('float')
    xB_star = b
    zN_star = -cN

    A = np.concatenate([N,B], axis=1)

    return N_index, B_index, A, b, cN, xB_star, zN_star


def optimize(N, b, cN):
    N_index, B_index, A, b, cN, xB_star, zN_star = preprocess(N, b, cN)
    if (xB_star>=0).all():
        status, solution, optimal_value, B, N, B_index, N_index, xB_star, zN_star, solution_str = primal(N_index=N_index, B_index=B_index, A=A, b=b, cN=cN, xB_star=xB_star, zN_star=zN_star)
    elif (zN_star>=0).all():
        status, solution, optimal_value, B, N, B_index, N_index, xB_star, zN_star, solution_str = dual(N_index=N_index, B_index=B_index, A=A, b=b, cN=cN, xB_star=xB_star, zN_star=zN_star)
    else:
        status, solution, optimal_value, B, N, B_index, N_index, xB_star, zN_star, solution_str = two_phase(N_index=N_index, B_index=B_index, A=A, b=b, cN=cN, xB_star=xB_star, zN_star=zN_star)

    return status, solution, optimal_value, B, N, B_index, N_index, xB_star, zN_star, solution_str
