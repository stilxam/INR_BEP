import sys
sys.path.append('/home/goshko/INR_BEP')

import jax
import jax.numpy as jnp
from inr_utils.images import load_image_as_array


parrot = load_image_as_array(
    '/home/goshko/INR_BEP/example_data/parrot.png')


def integer_lattice_mapping(data:jax.Array):
    """
    Implementation of the integer lattice mapping introduced in "Seeing INRs as Fourier Series"
    """

    def generate_B(N:int, dim_input:int)->jax.Array:
        """
        Generate the set B = {n in N_0 x Z^(d-1) | ||n||_∞ <= N} \ H
            where H excludes specific vectors 
        :parameter N: Maximum absolute value for each dimension
        :parameter dim_input: Dimension of the space
        :return: integer lattice mapping matrix B with shape (m, dim_input)
        """
        
        # Generate a grid of integer values for each dimension within the range [-N, N]
        # The shape of this grid is (2*N + 1)^dim_input.
        # This will create all possible combinations within the range [-N, N] for each dimension.
        grid_ranges = [jnp.arange(-N, N + 1) for _ in range(dim_input)]

        # jax.meshgrid to create a full grid across all dimensions.
        # meshgrid will create a separate array for each dimension in the grid_ranges,
        # where each array has all possible combinations of values in the other dimensions.
        grid_mesh = jnp.meshgrid(*grid_ranges, indexing='ij')

        # Flatten each dimension's array and stack them into an (m, dim_input) matrix.
        # - Each array in grid_mesh is flattened to a 1D array of length m.
        # - Then, these flattened arrays are stacked along the last dimension.
        # This gives us all possible coordinate combinations within the specified range.
        B = jnp.stack([g.flatten() for g in grid_mesh], axis=-1)

        # Filter out vectors that don't meet the infinity norm constraint.
        # The infinity norm of each vector (row) is the maximum absolute value of its components.
        # We keep only those rows where the infinity norm is <= N.
        mask = jnp.max(jnp.abs(B), axis=1) <= N
        B = B[mask]

        # Apply exclusion rule to remove vectors in set H
        # H excludes vectors where there exists an index j (2 <= j <= d) such that
        # all previous components n1 to n(j-1) are zero and nj < 0
        def in_H(vector):
            for j in range(1, dim_input):
                # Check if all previous components are zero
                if jnp.all(vector[:j] == 0):
                    # If the current component is negative, this vector belongs to H
                    if vector[j] < 0:
                        return True
            return False

        mask_H = jnp.array([not in_H(row) for row in B])
        B = B[mask_H]

        return B

