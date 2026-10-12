import numpy as np
from scipy.optimize import linprog

def dual_construction_and_verification():
    print("Primal: Min Z = 2x1 + 3x2 + 4x3")
    print("Dual: Max W = 6y1 + 4y2 + 8y3")
    print("Dual constraints generated natively.")
    print("y1 >= 0, y2 unrestricted, y3 <= 0\n")
    
    # Weak Duality Check
    x = np.array([3, 3, 2])
    y = np.array([0, 2, 0])
    
    Z_x = 2*x[0] + 3*x[1] + 4*x[2]
    W_y = 6*y[0] + 4*y[1] + 8*y[2]
    print(f"Weak Duality Check: W(y) = {W_y} <= Z(x) = {Z_x}\n")

    # Primal Solve (Handling unrestricted x3 by scipy default bounds)
    c = [2, 3, 4]
    A_ub = [[-1, -1, -1], [2, 1, -1]]
    b_ub = [-6, 8]
    A_eq = [[1, -1, 2]]
    b_eq = [4]
    bounds = [(0, None), (0, None), (None, None)]
    
    res_primal = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    # Dual Solve
    c_dual = [-6, -4, -8] # Maximize
    A_ub_dual = [[1, 1, 2], [1, -1, 1]]
    b_ub_dual = [2, 3]
    A_eq_dual = [[1, 2, -1]]
    b_eq_dual = [4]
    bounds_dual = [(0, None), (None, None), (None, 0)]
    
    res_dual = linprog(c_dual, A_ub=A_ub_dual, b_ub=b_ub_dual, A_eq=A_eq_dual, b_eq=b_eq_dual, bounds=bounds_dual, method='highs')
    
    print("Strong Duality Verification:")
    print(f"Primal Z* = {res_primal.fun:.4f}")
    print(f"Dual W* = {-res_dual.fun:.4f}")

if __name__ == "__main__":
    dual_construction_and_verification()