import numpy as np

def compute_mat_products(*matrices: np.ndarray) -> np.ndarray:
    """
    Multiplies a sequence of numpy arrays (matrices) using matrix multiplication.
    Args:
        *matrices: Variable number of numpy.ndarray objects to be multiplied.
    Returns:
        np.ndarray: The result of multiplying all input matrices in order.
    Raises:
        ValueError: If no matrices are provided, if any input is not a numpy.ndarray,
                    or if the matrices have incompatible dimensions for multiplication.
    """
    if not matrices:
        raise ValueError("At least one matrix must be provided.")
    for i, mat in enumerate(matrices):
        if not isinstance(mat, np.ndarray):
            raise TypeError(f"Argument {i+1} is not a numpy.ndarray.")
    result = matrices[0]
    for idx, mat in enumerate(matrices[1:], start=1):
        if result.shape[1] != mat.shape[0]:
            raise ValueError(f"Incompatible dimensions for multiplication between matrix {idx} and matrix {idx+1}: "
                             f"{result.shape} and {mat.shape}")
        result = result @ mat
    return result

def skew_symmetric(vec: np.ndarray) -> np.ndarray:
    """
    Return the 3x3 skew-symmetric matrix for a 3-element vector.

    Args:
        vec (np.ndarray): 1D array-like with 3 elements.

    Returns:
        np.ndarray: 3x3 skew-symmetric matrix S such that S @ w = vec x w.

    Raises:
        TypeError: if input cannot be converted to a numpy array.
        ValueError: if input does not have exactly 3 elements.
    """
    vec = np.asarray(vec)
    if vec.size != 3:
        raise ValueError("Input vector must have exactly 3 elements.")
    x, y, z = vec.ravel()
    return np.array([[ 0, -z,  y],
                     [ z,  0, -x],
                     [-y,  x,  0]])
