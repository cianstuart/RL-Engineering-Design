import numpy as np
from fem.truss_2d_parametric import Truss2DParametric

def test_fem_solver():
    truss = Truss2DParametric()

    x_m, y_m = 1.3, 0.2
    result = truss.solve(x_m, y_m)

    # Check fixed DOFs remain fixed 
    fixed_displacements = result.displacements[truss.fixed_dofs]
    is_fixed_ok = np.allclose(fixed_displacements, 0.0)
    print(f"Fixed support nodes: {'PASSED' if is_fixed_ok else 'FAILED'}")

    # Check whether the tip  moves downward 
    tip_u_y = result.displacements[5]
    is_deflection_ok = tip_u_y < 0.0
    print(f"Tip moves down under downward load: {'PASSED' if is_deflection_ok else 'FAILED'} (u_y = {tip_u_y:.6e} m)")

    # Check for positive compliance 
    is_compliance_ok = result.compliance > 0.0
    print(f"Structural compliance > 0: {'PASSED' if is_compliance_ok else 'FAILED'} (C = {result.compliance:.6f} J)")

    # Check for NaN in displacements and stresses
    no_nans = not np.isnan(result.displacements).any() and not np.isnan(result.stresses).any()
    print(f"No NaN numerical errors: {'PASSED' if no_nans else 'FAILED'}")

    # Print the stresses
    for idx, s in enumerate(result.stresses):
        print(f" Stress in bar {idx}: {s / 1e6:8.2f} MPa")

if __name__ == "__main__":
    test_fem_solver()