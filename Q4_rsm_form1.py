import numpy as np

def revised_simplex_q4():
    # Max Z = 3x1 + 2x2 + 5x3
    A = np.array([
        [1, 2, 1],
        [3, 0, 2],
        [1, 4, 0],
        [-3, -2, -5]
    ], dtype=float)
    
    b = np.array([430, 460, 420, 0], dtype=float)
    B_inv = np.eye(4)
    X_B = b.copy()
    
    basis = ['s1', 's2', 's3', 'Z']
    vars = ['x1', 'x2', 'x3']
    
    max_iters = 10
    for iteration in range(1, max_iters):
        deltas = B_inv[-1, :] @ A
        
        if np.all(deltas >= -1e-9):
            print(f"\nOptimal solution reached at iteration {iteration-1}.")
            break
            
        entering_idx = np.argmin(deltas)
        alpha = B_inv @ A[:, entering_idx]
        
        ratios = [X_B[i] / alpha[i] if alpha[i] > 0 else float('inf') for i in range(3)]
        leaving_idx = np.argmin(ratios)
        
        E = np.eye(4)
        eta = -alpha / alpha[leaving_idx]
        eta[leaving_idx] = 1 / alpha[leaving_idx]
        E[:, leaving_idx] = eta
        
        B_inv = E @ B_inv
        X_B = E @ X_B
        basis[leaving_idx] = vars[entering_idx]

    print("Final Basis:", basis)
    print("Final B_inv:\n", np.round(B_inv, 4))
    print("Final X_B (Solution):", np.round(X_B, 4))
    
if __name__ == "__main__":
    revised_simplex_q4()