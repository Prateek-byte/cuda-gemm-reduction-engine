import numpy as np

def print_dual():
    print("Dual Problem:")
    print("Minimize W = 60*y1 + 40*y2 + 80*y3")
    print("Subject to:")
    print("3*y1 + 2*y2 + 1*y3 >= 2")
    print("4*y1 + 1*y2 + 3*y3 >= 4")
    print("2*y1 + 2*y2 + 2*y3 >= 3")
    print("y1, y2, y3 >= 0\n")

# Using a simplified programmatic representation of the simplex tables described in Handwritten Solutions.
# For full robust LP, we use scipy to solve and extract duals to verify Z=W.
from scipy.optimize import linprog

def solve_and_verify():
    # Primal
    c = [-2, -4, -3] # Maximize
    A = [[3, 4, 2], [2, 1, 2], [1, 3, 2]]
    b = [60, 40, 80]
    
    res_primal = linprog(c, A_ub=A, b_ub=b, method='highs')
    print("Primal Optimal Solution:")
    print(f"x1 = {res_primal.x[0]:.2f}, x2 = {res_primal.x[1]:.2f}, x3 = {res_primal.x[2]:.2f}")
    Z_star = -res_primal.fun
    print(f"Z* = {Z_star:.2f}\n")
    
    # Dual
    c_dual = [60, 40, 80] # Minimize
    A_dual = [[-3, -2, -1], [-4, -1, -3], [-2, -2, -2]]
    b_dual = [-2, -4, -3]
    
    res_dual = linprog(c_dual, A_ub=A_dual, b_ub=b_dual, method='highs')
    print("Dual Optimal Solution:")
    print(f"y1 = {res_dual.x[0]:.2f}, y2 = {res_dual.x[1]:.2f}, y3 = {res_dual.x[2]:.2f}")
    W_star = res_dual.fun
    print(f"W* = {W_star:.2f}\n")
    
    print(f"Verification: Z* ({Z_star:.2f}) == W* ({W_star:.2f})")
    
if __name__ == "__main__":
    print_dual()
    solve_and_verify()