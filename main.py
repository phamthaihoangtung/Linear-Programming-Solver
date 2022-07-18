import streamlit as st
from utils import *
from linear_optimization import *


st.set_page_config(layout="wide")
st.title('Linear Programming Solver')

cols = st.columns(2)
with cols[0]:
    num_variables = st.number_input('Insert number of variables', min_value=1, step=1)
with cols[1]:
    num_constraints = st.number_input('Insert number of constraints', min_value=1, step=1)

# Objective function
st.subheader('Objective Function')

min_max, _ = st.columns((0.2, 0.8))
with min_max:
    min_max = st.selectbox('', ['max', 'min'])

obj_cols = st.columns(num_variables+2)

obj_coeff = [0 for i in range(num_variables)]
for i in range(num_variables):
    with obj_cols[i]:
        obj_coeff[i] = st.number_input(f'x{get_sub(str(i))}', key=f'obj_{i}', 
        # value=None,
        step=0.1,  
        format='%.1f'
        )
        # try:
        #     obj_coeff[i] = float(st.text_input(f'x{get_sub(str(i))}', key=f'obj_{i}', placeholder='0'))
        # except Exception as e:
        #     st.exception(e)


st.subheader('Constraints')
constraints_coeff = [[0 for i in range(num_variables)] for j in range(num_constraints)]
signs = ['' for i in range(num_constraints)]
# right hand side
rhs = [0 for i in range(num_constraints)]

for i in range(num_constraints):
    cstr_cols = st.columns(num_variables+2)
    for j in range(num_variables):
        with cstr_cols[j]:
            constraints_coeff[i][j] = st.number_input(f'x{get_sub(str(j))}', key=f'cstr_{i}{j}', 
            # value=None,
            step=0.1, 
            format='%.1f'
            )
    
    with cstr_cols[-2]:
        signs[i] = st.selectbox('', [u'\u2264', '=', u'\u2265'], key=f'sign_{i}')
    
    with cstr_cols[-1]:
        rhs[i] = st.number_input('', key=f'rhs_{i}', 
        # value=None, 
        step=0.1, 
        format='%.1f'
        )

x_cstr_string = ', '. join([f'x{get_sub(str(i))}' for i in range(num_variables)]) + ' ' + u'\u2265' + ' ' + '0'

st.subheader(x_cstr_string)

N, b, c = get_standard_form(min_max, obj_coeff, constraints_coeff, signs, rhs)

solve = st.button('Solve')

if solve:
    try:
        assert sum([coeff != 0 for coeff in obj_coeff]) != 0 # at least 1 coeff != 0
    except AssertionError as e:
        st.exception("Require at least 1 non-zero coefficient of objective function!")
        solve = False

    try:
        assert sum([sum([coeff != 0 for coeff in constraint])!=0 for constraint in constraints_coeff]) == len(constraints_coeff) # all row require have at least 1 coeff
    except AssertionError as e:
        st.exception("All constraint row require have at least 1 coefficient!")
        solve = False

if solve:
    print("Solving ...")

    status, solution, optimal_value, _, _, _, _, _, _, solution_str = optimize(N, b, c)
    
    tab1, tab2 = st.tabs(('Result', 'Steps'))

    with tab1:
        # st.subheader(status)
        if status == 'Optimal':
            st.subheader('Optimal solution')
            _, c = st.columns((0.2, 0.8))
            with c:
                for i in range(len(solution)):
                    st.subheader(f'x{get_sub(str(i))} = {solution[i]}')
            st.subheader('Optimal value')
            _, c = st.columns((0.2, 0.8))            
            c.subheader(str(optimal_value))

        else:
            st.subheader(f"The problem is {status}.")

    with tab2:
        st.code(solution_str)
