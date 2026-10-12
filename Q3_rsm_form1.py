import numpy as np

def revised_simplex():
    # Max Z = 2x1 + x2 -> Z - 2x1 - x2 = 0
    A = np.array([[3, 4], [6, 1], [-2, -1]], dtype=float)
    b = np.array([6, 3, 0], dtype=float)
    
    B_inv = np.eye(3)
    X_B = b.copy()
    basis = ['s1', 's2', 'Z']
    vars = ['x1', 'x2']
    
    for iteration in range(1, 4):
        print(f"--- Iteration {iteration} ---")
        print("B_inv:\n", B_inv)
        print("X_B:", X_B)
        
        # Net evaluations
        deltas = B_inv[-1, :] @ A
        print("Deltas:", deltas)
        
        if np.all(deltas >= -1e-9):
            print("Optimal solution reached.")
            break
            
        entering_idx = np.argmin(deltas)
        print(f"Entering variable: {vars[entering_idx]}")
        
        alpha = B_inv @ A[:, entering_idx]
        print("Alpha:", alpha)
        
        ratios = []
        for i in range(2): # Constraint rows only
            if alpha[i] > 0:
                ratios.append(X_B[i] / alpha[i])
            else:
                ratios.append(float('inf'))
                
        leaving_idx = np.argmin(ratios)
        print(f"Leaving variable: {basis[leaving_idx]}")
        
        # Form Elementary Matrix E
        E = np.eye(3)
        eta = -alpha / alpha[leaving_idx]
        eta[leaving_idx] = 1 / alpha[leaving_idx]
        E[:, leaving_idx] = eta
        
        B_inv = E @ B_inv
        X_B = E @ X_B
        basis[leaving_idx] = vars[entering_idx]

if __name__ == "__main__":
    revised_simplex()