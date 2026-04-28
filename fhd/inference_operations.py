import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
from scipy.optimize import lsq_linear

def crop_and_rescale(
    year,
    gdf,
    corop_gdf,
    corop_names,
    cols,
    gemeenten,
    cell_size = 500,
    padding_factor = 0, ):
    if year >= 2015 and year < 2017:
        index = 1
    if year >= 2017:  
        index = 0

    # --- 1. Get boundary ---
    boundary = corop_gdf[corop_gdf["statnaam"].isin(corop_names)]
    minx, miny, maxx, maxy = boundary.total_bounds

    width  = maxx - minx
    height = maxy - miny
    side = max(width, height)

    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    half = side / 2

    square_bounds = (
        cx - half/2,
        cy - half/2,
        cx + half,
        cy + half,
    )

    # --- 2. Optional padding ---
    padding = padding_factor * cell_size
    square_bounds = (
        square_bounds[0] - padding,
        square_bounds[1] - padding,
        square_bounds[2] + padding,
        square_bounds[3] + padding,
    )

    # --- 3. Crop ---
    clipped = gdf.cx[
        square_bounds[0]:square_bounds[2],
        square_bounds[1]:square_bounds[3]
    ].copy()

    # max_clipped = max_gdf.cx[
    #     square_bounds[0]:square_bounds[2],
    #     square_bounds[1]:square_bounds[3]
    # ]

    gemeenten_clipped = gemeenten.cx[
        square_bounds[0]:square_bounds[2],  
        square_bounds[1]:square_bounds[3]
    ]
    
    # --- 4. Add middle income if needed ---
    if (
        cols[2][index] not in clipped.columns
        and cols[1][index] in clipped.columns
        and cols[3][index] in clipped.columns
    ):
        clipped[cols[2][index]] = (
            100
            - clipped[cols[1][index]]
            - clipped[cols[3][index]]
        )

    # --- 5. Reduce columns ---
    keep_cols = [col[index] for col in cols if col[index] in clipped.columns]
    clipped = clipped[keep_cols + ["geometry"]].copy()

    # --- 6. Rescale ---
    clipped["percentage_leeg"] = (clipped[cols[5][index]] / clipped[cols[4][index]]) 
    
    clipped["percentage_vol"] = 1 - clipped["percentage_leeg"]
    

    for col in [
        cols[index][1],
        cols[index][2],
        cols[index][3],
    ]:
        if col in clipped.columns:
            clipped[f"{col}_rs"] = (
                clipped[col] * clipped["percentage_vol"]
            ) / 100

    return clipped, square_bounds, gemeenten_clipped

def grad_x(f, dx):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dx)

def grad_y(f, dy):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dy)

def laplacian(f, dx, dy):
    return (
        (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / dx**2 +
        (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / dy**2
    )

def divergence(fx, fy, dx, dy):
    return grad_x(fx, dx) + grad_y(fy, dy)

def biharmonic(f, dx, dy):
    return laplacian(laplacian(f, dx, dy), dx, dy)

def mirrorred(input):
    layers = input.copy()
    nan_mask = np.isnan(layers)   # shape: (3, ny, nx)

    for i in range(3):
        band = layers[i]
        mask = np.isnan(band)

        # indices of nearest non-NaN cell
        _, indices = distance_transform_edt(
            mask,
            return_indices=True
        )
        

        layers[i][mask] = band[tuple(indices[:, mask])]
    return layers, nan_mask

def build_A(rho, a, dx, dy, beta=1, include_nu=True, include_Gamma=True, scaling=False):
    # build the feature matrix A for species a
    ns, Nx, Ny = rho.shape
    rho0 = 1 - np.sum(rho, axis=0)
    assert ns == 3

    ra = rho[a]
    A = []

    # indices of the other species
    others = [i for i in range(ns) if i != a]
    b, c = others

    rb = rho[b]
    rc = rho[c]

    grad_ra = (grad_x(ra, dx), grad_y(ra, dy))
    grad_rb = (grad_x(rb, dx), grad_y(rb, dy))
    grad_rc = (grad_x(rc, dx), grad_y(rc, dy))
    grad_r0 = (grad_x(rho0, dx), grad_y(rho0, dy))

    rhoa0 = ra * rho0

    # --- 1. Diffusion ---
    fx = rho0 * grad_ra[0] - ra * grad_r0[0]
    fy = rho0 * grad_ra[1] - ra * grad_r0[1]
    A.append(divergence(fx, fy, dx, dy))

    # --- 2–4. Kappa ---
    for grad_r in [grad_ra, grad_rb, grad_rc]:
        fx = rhoa0 * grad_r[0]
        fy = rhoa0 * grad_r[1]
        A.append(-beta * divergence(fx, fy, dx, dy))  # minus from PDE

    # --- 5–10. Nu (reduced set) ---
    if include_nu:
        # ν^{aaa}
        fx = rhoa0 * (ra * grad_ra[0] + ra * grad_ra[0])
        fy = rhoa0 * (ra * grad_ra[1] + ra * grad_ra[1])
        A.append(-beta * divergence(fx, fy, dx, dy))

        # ν^{abb}
        fx = rhoa0 * (rb * grad_rb[0] + rb * grad_rb[0])
        fy = rhoa0 * (rb * grad_rb[1] + rb * grad_rb[1])
        A.append(-beta * divergence(fx, fy, dx, dy))

        # ν^{acc}
        fx = rhoa0 * (rc * grad_rc[0] + rc * grad_rc[0])
        fy = rhoa0 * (rc * grad_rc[1] + rc * grad_rc[1])
        A.append(-beta * divergence(fx, fy, dx, dy))

        # ν^a_ab = ν^{aab} + ν^{aba}
        fx = rhoa0 * (ra * grad_rb[0] + rb * grad_ra[0])
        fy = rhoa0 * (ra * grad_rb[1] + rb * grad_ra[1])
        A.append(-beta * divergence(fx, fy, dx, dy))

        # ν^a_ac
        fx = rhoa0 * (ra * grad_rc[0] + rc * grad_ra[0])
        fy = rhoa0 * (ra * grad_rc[1] + rc * grad_ra[1])
        A.append(-beta * divergence(fx, fy, dx, dy))

        # ν^a_bc
        fx = rhoa0 * (rb * grad_rc[0] + rc * grad_rb[0])
        fy = rhoa0 * (rb * grad_rc[1] + rc * grad_rb[1])
        A.append(-beta * divergence(fx, fy, dx, dy))
    else:
        A.append(np.zeros_like(ra))  # placeholder for ν^{aaa}
        A.append(np.zeros_like(ra))  # placeholder for ν^{abb}
        A.append(np.zeros_like(ra))  # placeholder for ν^{acc}
        A.append(np.zeros_like(ra))  # placeholder for ν^a_ab
        A.append(np.zeros_like(ra))  # placeholder for ν^a_ac
        A.append(np.zeros_like(ra))  # placeholder for ν^a_bc

    if include_Gamma:
        # --- 11. Gamma ---
        lap_ra = laplacian(ra, dx, dy)
        grad_lap_ra = (grad_x(lap_ra, dx), grad_y(lap_ra, dy))

        fx = rhoa0 * grad_lap_ra[0]
        fy = rhoa0 * grad_lap_ra[1]
        A.append(-beta * divergence(fx, fy, dx, dy))
    else:
        A.append(np.zeros_like(ra))  # placeholder for Gamma

    # --- stack into matrix ---
    A = np.stack([f.reshape(-1) for f in A], axis=1)

    if scaling:
        # --- normalize ---
        scales = np.std(A, axis=0) + 1e-12
        A /= scales
        return A, scales

    return A, None

def unpack_theta(Thetas, ns=3):
    """
    Convert list of theta_a (length 11 each) into:
        D:     (ns,)
        kappa: (ns, ns)
        nu:    (ns, ns, ns)   # nu[a, b, c]
        Gamma: (ns, ns)       # each row a filled with Gamma^a (repeated)
    """
    assert ns == 3, "This implementation assumes ns=3"
    assert len(Thetas) == ns

    D = np.zeros(ns)
    kappa = np.zeros((ns, ns))
    nu = np.zeros((ns, ns, ns))
    Gamma = np.zeros((ns, ns))

    for a in range(ns):
        theta = Thetas[a]
        assert len(theta) == 11

        # indices of the other species
        others = [i for i in range(ns) if i != a]
        b, c = others

        # --- 1. D ---
        D[a] = theta[0]

        # --- 2. kappa ---
        # order: κ_aa, κ_ab, κ_ac
        kappa[a, a] = theta[1]
        kappa[a, b] = theta[2]
        kappa[a, c] = theta[3]

        # --- 3. nu (fill symmetric in b,c) ---
        # order:
        # ν_aaa, ν_abb, ν_acc, ν^a_ab, ν^a_ac, ν^a_bc

        # diagonal blocks
        nu[a, a, a] = theta[4]
        nu[a, b, b] = theta[5]
        nu[a, c, c] = theta[6]

        # mixed terms (split symmetric pairs)
        nu[a, a, b] = nu[a, b, a] = theta[7] / 2
        nu[a, a, c] = nu[a, c, a] = theta[8] / 2
        nu[a, b, c] = nu[a, c, b] = theta[9] / 2

        # --- 4. Gamma ---
        # repeat Gamma^a across row a (your requested format)
        Gamma[a, :] = theta[10]

    return D, kappa, nu, Gamma

def compute_thetas(mirror_start, mirror_end, nan_mask_start, dx, dy, ns, delta_t, include_nu=True, include_Gamma=True, scaling=True):
    # Compute time derivative
    partial_m = np.zeros_like(mirror_start)
    for i in range(mirror_start.shape[0]):
        partial_m[i] = (mirror_end[i] - mirror_start[i]) / delta_t

    thetas = []
    stds = []

    for a in range(ns):
        drho_dt_a = partial_m[a].reshape(-1)

        A, scales = build_A(
            mirror_start, a, dx, dy,
            include_nu=True, include_Gamma=True, scaling=True
        )

        nan_mask_flat = nan_mask_start[a].reshape(-1)
        drho_dt_a = drho_dt_a[~nan_mask_flat]
        A = A[~nan_mask_flat]

        stds.append(np.std(A, axis=0))

        # Bounds: first parameter >= 0, others free
        n_params = A.shape[1]
        lb = np.full(n_params, -np.inf)
        ub = np.full(n_params,  np.inf)
        lb[0] = 0.0

        res = lsq_linear(A, drho_dt_a, bounds=(lb, ub))
        theta_a = res.x

        if scales is not None:
            theta_a = theta_a * scales

        thetas.append(theta_a)

    return thetas, stds