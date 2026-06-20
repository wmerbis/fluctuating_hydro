import numpy as np

#-----------------------------Jay's prediction-------------------------------
# === Drift matrix M(k) and noise matrix Q(k) ===
def noise_matrix(k, a0, b0, param):
    """Return the 2x2 noise covariance Q(k) = Q0 + Q2 k^2 in 2D."""
    DA, DB = param["D"]
    Dv = param["D_v"]
    h = param['h']
    k2 = k**2
    r0 = 1-a0-b0
    q11 = DA * h**2 * r0 * a0 * k2 + 2.0 * Dv * a0* b0
    q22 = DB * h**2 * r0 * b0 * k2 + 2.0 * Dv * a0 * b0
    q12 = -2.0 * Dv * a0* b0
    return np.array([[q11, q12],
                     [q12, q22]], dtype=float)

def drift_matrix(k, a0, b0, param):
    """Return the 2x2 drift matrix M(k) = A2 k^2 + A4 k^4."""
    k2 = k**2
    k4 = k2**2
    r0 = 1-a0-b0
    DA, DB = param["D"]
    Dv = param["D_v"]
    betaA = param["beta"]
    betaB = param["beta"]
    Gamma_aa = param["Gamma"][0,0]
    Gamma_ab = param["Gamma"][0,1]
    Gamma_ba = param["Gamma"][1,0]
    Gamma_bb = param["Gamma"][1,1]
    kappa_aa = param["kappa"][0,0]
    kappa_ab = param["kappa"][0,1]
    kappa_ba = param["kappa"][1,0]
    kappa_bb = param["kappa"][1,1]
    
    Maa = (DA*(r0 + a0) + Dv*b0 - DA*betaA*r0*a0*kappa_aa) * k2 + DA*betaA*r0*a0*Gamma_aa * k4
    
    Mab = (DA*a0 - Dv*a0 - DA*betaA*r0*a0*kappa_ab) * k2 + DA*betaA*r0*a0*Gamma_ab * k4
    
    Mba = (DB*b0 - Dv*b0 - DB*betaB*r0*b0*kappa_ba) * k2 + DB*betaB*r0*b0*Gamma_ba * k4
    
    Mbb = (DB*(r0 + b0) + Dv*a0 - DB*betaB*r0*b0*kappa_bb) * k2 + DB*betaB*r0*b0*Gamma_bb * k4
    
    return np.array([[Maa, Mab],
                     [Mba, Mbb]], dtype=float)

def solve_lyapunov_2x2(M, R):
    """Solve M S + S M^T = R for symmetric 2x2 S.
    
    M = [[a,b],[c,d]]
    S = [[x,y],[y,z]]
    R = [[R11,R12],[R12,R22]]
    """
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    R11, R12, R22 = R[0, 0], R[0, 1], R[1, 1]
    
    # Linear system:
    # a x + b y = R11/2
    # c y + d z = R22/2
    # c x + (a + d) y + b z = R12
    A = np.array([
        [a,     b,   0.0],
        [0.0,   c,   d  ],
        [c, a + d,   b  ]
    ], dtype=float)
    rhs = np.array([R11/2.0, R22/2.0, R12], dtype=float)
    
    x, y, z = np.linalg.solve(A, rhs)
    return np.array([[x, y],
                     [y, z]], dtype=float)


def structure_factor(k, a0, b0, param):
    """Full equal-time structure factor S(k) solving M(k) S + S M^T = 2 Q(k)."""
    M = drift_matrix(k, a0, b0, param)
    Q = noise_matrix(k, a0, b0, param)
    R = 2.0 * Q
    return solve_lyapunov_2x2(M, R)

# def SAA(k, a0, b0, h):
#     return structure_factor(k, a0, b0, h)[0, 0]


# def SAB(k, a0, b0, h):
#     return structure_factor(k, a0, b0, h)[0, 1]


# def SBB(k, a0, b0, h):
#     return structure_factor(k, a0, b0, h)[1, 1]