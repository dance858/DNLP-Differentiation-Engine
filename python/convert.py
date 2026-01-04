import cvxpy as cp
import numpy as np
from scipy import sparse
from cvxpy.reductions.inverse_data import InverseData
import DNLP_diff_engine as diffengine


def get_jacobian(c_expr, values):
    """Compute jacobian and return as scipy sparse CSR matrix."""
    data, indices, indptr, shape = diffengine.jacobian(c_expr, values)
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def _chain_add(children):
    """Chain multiple children with binary adds: a + b + c -> add(add(a, b), c)."""
    result = children[0]
    for child in children[1:]:
        result = diffengine.make_add(result, child)
    return result


# Mapping from CVXPY atom names to C diff engine functions
ATOM_CONVERTERS = {
    # Elementwise unary
    "log": lambda child: diffengine.make_log(child),
    "exp": lambda child: diffengine.make_exp(child),

    # N-ary (handles 2+ args)
    "AddExpression": _chain_add,

    # Reductions
    "Sum": lambda child: diffengine.make_sum(child, -1),  # axis=-1 = sum all
}


def build_variable_dict(variables: list) -> tuple[dict, int]:
    """
    Build dictionary mapping CVXPY variable ids to C variables.

    Args:
        variables: list of CVXPY Variable objects

    Returns:
        var_dict: {var.id: c_variable} mapping
        n_vars: total number of scalar variables
    """
    id_map, _, n_vars, var_shapes = InverseData.get_var_offsets(variables)

    var_dict = {}
    for var in variables:
        offset, _ = id_map[var.id]
        shape = var_shapes[var.id]
        if len(shape) == 2:
            d1, d2 = shape[0], shape[1]
        elif len(shape) == 1:
            d1, d2 = shape[0], 1
        else:  # scalar
            d1, d2 = 1, 1
        c_var = diffengine.make_variable(d1, d2, offset, n_vars)
        var_dict[var.id] = c_var

    return var_dict


def _convert_expr(expr, var_dict: dict):
    """Convert CVXPY expression using pre-built variable dictionary."""
    # Base case: variable lookup
    if isinstance(expr, cp.Variable):
        return var_dict[expr.id]

    # Recursive case: atoms
    atom_name = type(expr).__name__
    if atom_name in ATOM_CONVERTERS:
        children = [_convert_expr(arg, var_dict) for arg in expr.args]
        converter = ATOM_CONVERTERS[atom_name]
        # N-ary ops (like AddExpression) take list, unary ops take single arg
        if atom_name == "AddExpression":
            return converter(children)
        return converter(*children) if len(children) > 1 else converter(children[0])

    raise NotImplementedError(f"Atom '{atom_name}' not supported")


def convert_problem(problem: cp.Problem) -> tuple:
    """
    Convert CVXPY Problem to C expressions.

    Args:
        problem: CVXPY Problem object

    Returns:
        c_objective: C expression for objective
        c_constraints: list of C expressions for constraints
    """
    var_dict = build_variable_dict(problem.variables())

    # Convert objective
    c_objective = _convert_expr(problem.objective.expr, var_dict)

    # Convert constraints (expression part only for now)
    c_constraints = []
    for constr in problem.constraints:
        c_expr = _convert_expr(constr.expr, var_dict)
        c_constraints.append(c_expr)

    return c_objective, c_constraints


# === Tests ===

def test_simple_sum_log():
    """Test converting cp.sum(cp.log(x))."""
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([1.0, 2.0, 3.0])
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_simple_sum_log passed")


def test_two_variables():
    """Test problem with two variables: sum(log(x + y))."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x + y))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([1.0, 2.0, 3.0, 4.0])
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(np.array([1+3, 2+4])))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_two_variables passed")


def test_variable_reuse():
    """Test that same variable used twice works correctly."""
    x = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x) + cp.exp(x))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([1.0, 2.0])
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values) + np.exp(test_values))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_variable_reuse passed")


def test_four_variables():
    """Test problem with 4 variables: sum(log(a + b) + exp(c + d))."""
    a = cp.Variable(3)
    b = cp.Variable(3)
    c = cp.Variable(3)
    d = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(a + b) + cp.exp(c + d))))
    c_obj, _ = convert_problem(problem)

    a_vals = np.array([1.0, 2.0, 3.0])
    b_vals = np.array([0.5, 1.0, 1.5])
    c_vals = np.array([0.1, 0.2, 0.3])
    d_vals = np.array([0.1, 0.1, 0.1])
    test_values = np.concatenate([a_vals, b_vals, c_vals, d_vals])

    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(a_vals + b_vals) + np.exp(c_vals + d_vals))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_four_variables passed")


def test_deep_nesting():
    """Test deeply nested composition: sum(log(exp(log(exp(x)))))."""
    x = cp.Variable(4)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(cp.exp(cp.log(cp.exp(x)))))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([0.5, 1.0, 1.5, 2.0])
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(np.exp(np.log(np.exp(test_values)))))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_deep_nesting passed")


def test_chained_additions():
    """Test multiple chained additions: sum(x + y + z + w)."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    z = cp.Variable(2)
    w = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(x + y + z + w)))
    c_obj, _ = convert_problem(problem)

    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([3.0, 4.0])
    z_vals = np.array([5.0, 6.0])
    w_vals = np.array([7.0, 8.0])
    test_values = np.concatenate([x_vals, y_vals, z_vals, w_vals])

    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(x_vals + y_vals + z_vals + w_vals)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_chained_additions passed")


def test_variable_used_multiple_times():
    """Test variable used 3+ times: sum(log(x) + exp(x) + x)."""
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x) + cp.exp(x) + x)))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([1.0, 2.0, 3.0])
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values) + np.exp(test_values) + test_values)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_variable_used_multiple_times passed")


def test_larger_variable_size():
    """Test with larger variable (100 elements)."""
    x = cp.Variable(100)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(cp.exp(x)))))
    c_obj, _ = convert_problem(problem)

    test_values = np.linspace(0.1, 2.0, 100)
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(np.exp(test_values)))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_larger_variable_size passed")


def test_matrix_variable():
    """Test with 2D matrix variable (3x4)."""
    X = cp.Variable((3, 4))
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(X))))
    c_obj, _ = convert_problem(problem)

    test_values = np.arange(1.0, 13.0)  # 12 elements
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(test_values))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_matrix_variable passed")


def test_mixed_sizes():
    """Test with variables of different sizes."""
    a = cp.Variable(2)
    b = cp.Variable(5)
    c = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(a)) + cp.sum(cp.log(b)) + cp.sum(cp.log(c))))
    c_obj, _ = convert_problem(problem)

    a_vals = np.array([1.0, 2.0])
    b_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c_vals = np.array([1.0, 2.0, 3.0])
    test_values = np.concatenate([a_vals, b_vals, c_vals])

    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(np.log(a_vals)) + np.sum(np.log(b_vals)) + np.sum(np.log(c_vals))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_mixed_sizes passed")


def test_complex_objective():
    """Test complex objective: sum(log(x + y)) + sum(exp(y + z)) + sum(log(z + x))."""
    x = cp.Variable(3)
    y = cp.Variable(3)
    z = cp.Variable(3)
    obj = cp.sum(cp.log(x + y)) + cp.sum(cp.exp(y + z)) + cp.sum(cp.log(z + x))
    problem = cp.Problem(cp.Minimize(obj))
    c_obj, _ = convert_problem(problem)

    x_vals = np.array([1.0, 2.0, 3.0])
    y_vals = np.array([0.5, 1.0, 1.5])
    z_vals = np.array([0.2, 0.3, 0.4])
    test_values = np.concatenate([x_vals, y_vals, z_vals])

    result = diffengine.forward(c_obj, test_values)
    expected = (np.sum(np.log(x_vals + y_vals)) +
                np.sum(np.exp(y_vals + z_vals)) +
                np.sum(np.log(z_vals + x_vals)))
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_complex_objective passed")


def test_log_exp_identity():
    """Test log(exp(x)) = x identity."""
    x = cp.Variable(5)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(cp.exp(x)))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    result = diffengine.forward(c_obj, test_values)
    expected = np.sum(test_values)  # log(exp(x)) = x
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_log_exp_identity passed")


# === Jacobian Tests ===

def test_jacobian_sum_log():
    """Test jacobian of sum(log(x)). Gradient is [1/x_1, 1/x_2, ...]."""
    x = cp.Variable(4)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([1.0, 2.0, 3.0, 4.0])
    jac = get_jacobian(c_obj, test_values)

    # sum(log(x)) is scalar, so jacobian is 1 x n
    expected = (1.0 / test_values).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected), f"Expected {expected}, got {jac.toarray()}"
    print("test_jacobian_sum_log passed")


def test_jacobian_sum_exp():
    """Test jacobian of sum(exp(x)). Gradient is [exp(x_1), exp(x_2), ...]."""
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.exp(x))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([0.0, 1.0, 2.0])
    jac = get_jacobian(c_obj, test_values)

    expected = np.exp(test_values).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected), f"Expected {expected}, got {jac.toarray()}"
    print("test_jacobian_sum_exp passed")


def test_jacobian_four_variables():
    """Test jacobian of sum(log(a)) + sum(log(b)) + sum(exp(c)) + sum(exp(d))."""
    a = cp.Variable(2)
    b = cp.Variable(2)
    c = cp.Variable(2)
    d = cp.Variable(2)
    obj = cp.sum(cp.log(a)) + cp.sum(cp.log(b)) + cp.sum(cp.exp(c)) + cp.sum(cp.exp(d))
    problem = cp.Problem(cp.Minimize(obj))
    c_obj, _ = convert_problem(problem)

    # Variables: [a0, a1, b0, b1, c0, c1, d0, d1]
    a_vals = np.array([1.0, 2.0])
    b_vals = np.array([0.5, 1.0])
    c_vals = np.array([0.1, 0.2])
    d_vals = np.array([0.1, 0.1])
    test_values = np.concatenate([a_vals, b_vals, c_vals, d_vals])
    jac = get_jacobian(c_obj, test_values)

    # f = sum(log(a)) + sum(log(b)) + sum(exp(c)) + sum(exp(d))
    df_da = 1.0 / a_vals
    df_db = 1.0 / b_vals
    df_dc = np.exp(c_vals)
    df_dd = np.exp(d_vals)
    expected = np.concatenate([df_da, df_db, df_dc, df_dd]).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected), f"Expected {expected}, got {jac.toarray()}"
    print("test_jacobian_four_variables passed")


def test_jacobian_two_variables():
    """Test jacobian of sum(log(x) + log(y)) with two variables."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x) + cp.log(y))))
    c_obj, _ = convert_problem(problem)

    # Variables: [x0, x1, y0, y1]
    test_values = np.array([1.0, 2.0, 3.0, 4.0])
    jac = get_jacobian(c_obj, test_values)

    # f = sum(log(x) + log(y)) = log(x0) + log(x1) + log(y0) + log(y1)
    # df/dx = [1/x0, 1/x1], df/dy = [1/y0, 1/y1]
    expected = np.array([[1/1, 1/2, 1/3, 1/4]])
    assert np.allclose(jac.toarray(), expected), f"Expected {expected}, got {jac.toarray()}"
    print("test_jacobian_two_variables passed")


def test_jacobian_variable_reuse():
    """Test jacobian when same variable is used multiple times."""
    x = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x) + cp.exp(x))))
    c_obj, _ = convert_problem(problem)

    test_values = np.array([1.0, 2.0])
    jac = get_jacobian(c_obj, test_values)

    # f = sum(log(x) + exp(x))
    # df/dx_i = 1/x_i + exp(x_i)
    expected = (1.0 / test_values + np.exp(test_values)).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected), f"Expected {expected}, got {jac.toarray()}"
    print("test_jacobian_variable_reuse passed")


def test_jacobian_large_variable():
    """Test jacobian of sum(log(x)) with larger variable."""
    x = cp.Variable(10)
    problem = cp.Problem(cp.Minimize(cp.sum(cp.log(x))))
    c_obj, _ = convert_problem(problem)

    test_values = np.linspace(1.0, 10.0, 10)
    jac = get_jacobian(c_obj, test_values)

    # f = sum(log(x)), df/dx_i = 1/x_i
    expected = (1.0 / test_values).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected), f"Expected {expected}, got {jac.toarray()}"
    print("test_jacobian_large_variable passed")


def test_jacobian_complex_objective():
    """Test jacobian of sum(log(x) + exp(y) + log(z))."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    z = cp.Variable(2)
    obj = cp.sum(cp.log(x) + cp.exp(y) + cp.log(z))
    problem = cp.Problem(cp.Minimize(obj))
    c_obj, _ = convert_problem(problem)

    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([0.5, 1.0])
    z_vals = np.array([0.2, 0.3])
    test_values = np.concatenate([x_vals, y_vals, z_vals])
    jac = get_jacobian(c_obj, test_values)

    # f = sum(log(x) + exp(y) + log(z))
    # df/dx_i = 1/x_i, df/dy_i = exp(y_i), df/dz_i = 1/z_i
    df_dx = 1.0 / x_vals
    df_dy = np.exp(y_vals)
    df_dz = 1.0 / z_vals
    expected = np.concatenate([df_dx, df_dy, df_dz]).reshape(1, -1)
    assert np.allclose(jac.toarray(), expected), f"Expected {expected}, got {jac.toarray()}"
    print("test_jacobian_complex_objective passed")


if __name__ == "__main__":
    # Forward pass tests
    test_simple_sum_log()
    test_two_variables()
    test_variable_reuse()
    test_four_variables()
    test_deep_nesting()
    test_chained_additions()
    test_variable_used_multiple_times()
    test_larger_variable_size()
    test_matrix_variable()
    test_mixed_sizes()
    test_complex_objective()
    test_log_exp_identity()

    # Jacobian tests
    test_jacobian_sum_log()
    test_jacobian_sum_exp()
    test_jacobian_four_variables()
    test_jacobian_two_variables()
    test_jacobian_variable_reuse()
    test_jacobian_large_variable()
    test_jacobian_complex_objective()

    print("\nAll tests passed!")
