import cvxpy as cp
import numpy as np
from scipy import sparse
import DNLP_diff_engine as diffengine
from convert import Problem


# ============ Low-level problem struct tests ============

def test_problem_forward_lowlevel():
    """Test problem_forward for objective and constraint values (low-level)."""
    n_vars = 3
    x_obj = diffengine.make_variable(n_vars, 1, 0, n_vars)
    log_obj = diffengine.make_log(x_obj)
    objective = diffengine.make_sum(log_obj, -1)

    x_c = diffengine.make_variable(n_vars, 1, 0, n_vars)
    log_c = diffengine.make_log(x_c)
    constraints = [log_c]

    prob = diffengine.make_problem(objective, constraints)
    u = np.array([1.0, 2.0, 3.0])
    diffengine.problem_allocate(prob, u)

    obj_val, constraint_vals = diffengine.problem_forward(prob, u)

    expected_obj = np.sum(np.log(u))
    assert np.allclose(obj_val, expected_obj)
    assert np.allclose(constraint_vals, np.log(u))


def test_problem_gradient_lowlevel():
    """Test problem_gradient for objective gradient (low-level)."""
    n_vars = 3
    x = diffengine.make_variable(n_vars, 1, 0, n_vars)
    log_x = diffengine.make_log(x)
    objective = diffengine.make_sum(log_x, -1)

    prob = diffengine.make_problem(objective, [])
    u = np.array([1.0, 2.0, 4.0])
    diffengine.problem_allocate(prob, u)

    grad = diffengine.problem_gradient(prob, u)
    expected_grad = 1.0 / u
    assert np.allclose(grad, expected_grad)


def test_problem_jacobian_lowlevel():
    """Test problem_jacobian for constraint jacobian (low-level)."""
    n_vars = 2
    x_obj = diffengine.make_variable(n_vars, 1, 0, n_vars)
    log_obj = diffengine.make_log(x_obj)
    objective = diffengine.make_sum(log_obj, -1)

    x_c = diffengine.make_variable(n_vars, 1, 0, n_vars)
    log_c = diffengine.make_log(x_c)
    constraints = [log_c]

    prob = diffengine.make_problem(objective, constraints)
    u = np.array([2.0, 4.0])
    diffengine.problem_allocate(prob, u)

    data, indices, indptr, shape = diffengine.problem_jacobian(prob, u)
    jac = sparse.csr_matrix((data, indices, indptr), shape=shape)

    expected_jac = np.diag(1.0 / u)
    assert np.allclose(jac.toarray(), expected_jac)


def test_problem_no_constraints_lowlevel():
    """Test Problem with no constraints (low-level)."""
    n_vars = 3
    x = diffengine.make_variable(n_vars, 1, 0, n_vars)
    log_x = diffengine.make_log(x)
    objective = diffengine.make_sum(log_x, -1)

    prob = diffengine.make_problem(objective, [])
    u = np.array([1.0, 2.0, 3.0])
    diffengine.problem_allocate(prob, u)

    obj_val, constraint_vals = diffengine.problem_forward(prob, u)
    assert np.allclose(obj_val, np.sum(np.log(u)))
    assert len(constraint_vals) == 0

    grad = diffengine.problem_gradient(prob, u)
    assert np.allclose(grad, 1.0 / u)

    data, indices, indptr, shape = diffengine.problem_jacobian(prob, u)
    jac = sparse.csr_matrix((data, indices, indptr), shape=shape)
    assert jac.shape == (0, 3)


# ============ Problem class tests using convert ============

def test_problem_single_constraint():
    """Test Problem class with single constraint using convert."""
    x = cp.Variable(3)
    obj = cp.sum(cp.log(x))
    constraints = [cp.log(x)]  # log(x) as constraint expression

    cvxpy_prob = cp.Problem(cp.Minimize(obj), constraints)
    prob = Problem(cvxpy_prob)

    u = np.array([1.0, 2.0, 3.0])
    prob.allocate(u)

    # Test forward
    obj_val, constraint_vals = prob.forward(u)
    assert np.allclose(obj_val, np.sum(np.log(u)))
    assert np.allclose(constraint_vals, np.log(u))

    # Test gradient
    grad = prob.gradient(u)
    assert np.allclose(grad, 1.0 / u)

    # Test jacobian
    jac = prob.jacobian(u)
    expected_jac = np.diag(1.0 / u)
    assert np.allclose(jac.toarray(), expected_jac)


def test_problem_two_constraints():
    """Test Problem class with two constraints."""
    x = cp.Variable(2)
    obj = cp.sum(cp.log(x))
    constraints = [
        cp.log(x),  # constraint 1: log(x), size 2
        cp.exp(x),  # constraint 2: exp(x), size 2
    ]

    cvxpy_prob = cp.Problem(cp.Minimize(obj), constraints)
    prob = Problem(cvxpy_prob)

    u = np.array([1.0, 2.0])
    prob.allocate(u)

    # Test forward
    obj_val, constraint_vals = prob.forward(u)
    assert np.allclose(obj_val, np.sum(np.log(u)))
    # Constraint values should be stacked: [log(1), log(2), exp(1), exp(2)]
    expected_constraint_vals = np.concatenate([np.log(u), np.exp(u)])
    assert np.allclose(constraint_vals, expected_constraint_vals)

    # Test gradient
    grad = prob.gradient(u)
    assert np.allclose(grad, 1.0 / u)

    # Test jacobian - stacked vertically
    jac = prob.jacobian(u)
    assert jac.shape == (4, 2)  # 2 constraints * 2 elements each
    # First 2 rows: d(log(x))/dx = diag(1/x)
    # Last 2 rows: d(exp(x))/dx = diag(exp(x))
    expected_jac = np.vstack([np.diag(1.0 / u), np.diag(np.exp(u))])
    assert np.allclose(jac.toarray(), expected_jac)


def test_problem_three_constraints_different_sizes():
    """Test Problem with three constraints of different types."""
    x = cp.Variable(3)
    obj = cp.sum(cp.exp(x))
    constraints = [
        cp.log(x),       # size 3
        cp.exp(x),       # size 3
        cp.sum(cp.log(x)),  # size 1 (scalar)
    ]

    cvxpy_prob = cp.Problem(cp.Minimize(obj), constraints)
    prob = Problem(cvxpy_prob)

    u = np.array([1.0, 2.0, 3.0])
    prob.allocate(u)

    # Test forward
    obj_val, constraint_vals = prob.forward(u)
    assert np.allclose(obj_val, np.sum(np.exp(u)))
    # Constraint values: [log(x), exp(x), sum(log(x))]
    expected_constraint_vals = np.concatenate([
        np.log(u),
        np.exp(u),
        [np.sum(np.log(u))]
    ])
    assert np.allclose(constraint_vals, expected_constraint_vals)

    # Test gradient of sum(exp(x))
    grad = prob.gradient(u)
    assert np.allclose(grad, np.exp(u))

    # Test jacobian
    jac = prob.jacobian(u)
    assert jac.shape == (7, 3)  # 3 + 3 + 1 rows


def test_problem_multiple_variables():
    """Test Problem with multiple CVXPY variables."""
    x = cp.Variable(2)
    y = cp.Variable(2)
    obj = cp.sum(cp.log(x)) + cp.sum(cp.exp(y))
    constraints = [
        cp.log(x),
        cp.exp(y),
    ]

    cvxpy_prob = cp.Problem(cp.Minimize(obj), constraints)
    prob = Problem(cvxpy_prob)

    x_vals = np.array([1.0, 2.0])
    y_vals = np.array([0.5, 1.0])
    u = np.concatenate([x_vals, y_vals])
    prob.allocate(u)

    # Test forward
    obj_val, constraint_vals = prob.forward(u)
    expected_obj = np.sum(np.log(x_vals)) + np.sum(np.exp(y_vals))
    assert np.allclose(obj_val, expected_obj)
    expected_constraint_vals = np.concatenate([np.log(x_vals), np.exp(y_vals)])
    assert np.allclose(constraint_vals, expected_constraint_vals)

    # Test gradient
    grad = prob.gradient(u)
    expected_grad = np.concatenate([1.0 / x_vals, np.exp(y_vals)])
    assert np.allclose(grad, expected_grad)

    # Test jacobian
    jac = prob.jacobian(u)
    assert jac.shape == (4, 4)
    # First constraint log(x) only depends on x (first 2 vars)
    # Second constraint exp(y) only depends on y (last 2 vars)
    expected_jac = np.zeros((4, 4))
    expected_jac[0, 0] = 1.0 / x_vals[0]
    expected_jac[1, 1] = 1.0 / x_vals[1]
    expected_jac[2, 2] = np.exp(y_vals[0])
    expected_jac[3, 3] = np.exp(y_vals[1])
    assert np.allclose(jac.toarray(), expected_jac)


def test_problem_no_constraints_convert():
    """Test Problem class with no constraints using convert."""
    x = cp.Variable(4)
    obj = cp.sum(cp.log(x))

    cvxpy_prob = cp.Problem(cp.Minimize(obj))
    prob = Problem(cvxpy_prob)

    u = np.array([1.0, 2.0, 3.0, 4.0])
    prob.allocate(u)

    obj_val, constraint_vals = prob.forward(u)
    assert np.allclose(obj_val, np.sum(np.log(u)))
    assert len(constraint_vals) == 0

    grad = prob.gradient(u)
    assert np.allclose(grad, 1.0 / u)

    jac = prob.jacobian(u)
    assert jac.shape == (0, 4)


def test_problem_larger_scale():
    """Test Problem with larger variables and multiple constraints."""
    n = 50
    x = cp.Variable(n)
    obj = cp.sum(cp.log(x))
    constraints = [
        cp.log(x),
        cp.exp(x),
        cp.sum(cp.log(x)),
        cp.sum(cp.exp(x)),
    ]

    cvxpy_prob = cp.Problem(cp.Minimize(obj), constraints)
    prob = Problem(cvxpy_prob)

    u = np.linspace(1.0, 5.0, n)
    prob.allocate(u)

    # Test forward
    obj_val, constraint_vals = prob.forward(u)
    assert np.allclose(obj_val, np.sum(np.log(u)))
    expected_constraint_vals = np.concatenate([
        np.log(u),
        np.exp(u),
        [np.sum(np.log(u))],
        [np.sum(np.exp(u))],
    ])
    assert np.allclose(constraint_vals, expected_constraint_vals)

    # Test gradient
    grad = prob.gradient(u)
    assert np.allclose(grad, 1.0 / u)

    # Test jacobian shape
    jac = prob.jacobian(u)
    assert jac.shape == (n + n + 1 + 1, n)  # 102 x 50


def test_problem_repeated_evaluations():
    """Test Problem with repeated evaluations at different points."""
    x = cp.Variable(3)
    obj = cp.sum(cp.log(x))
    constraints = [cp.exp(x)]

    cvxpy_prob = cp.Problem(cp.Minimize(obj), constraints)
    prob = Problem(cvxpy_prob)

    u1 = np.array([1.0, 2.0, 3.0])
    prob.allocate(u1)

    # First evaluation
    obj_val1, _ = prob.forward(u1)
    grad1 = prob.gradient(u1)

    # Second evaluation at different point
    u2 = np.array([2.0, 3.0, 4.0])
    obj_val2, _ = prob.forward(u2)
    grad2 = prob.gradient(u2)

    assert np.allclose(obj_val1, np.sum(np.log(u1)))
    assert np.allclose(obj_val2, np.sum(np.log(u2)))
    assert np.allclose(grad1, 1.0 / u1)
    assert np.allclose(grad2, 1.0 / u2)


if __name__ == "__main__":
    # Low-level tests
    test_problem_forward_lowlevel()
    test_problem_gradient_lowlevel()
    test_problem_jacobian_lowlevel()
    test_problem_no_constraints_lowlevel()
    # Problem class tests
    test_problem_single_constraint()
    test_problem_two_constraints()
    test_problem_three_constraints_different_sizes()
    test_problem_multiple_variables()
    test_problem_no_constraints_convert()
    test_problem_larger_scale()
    test_problem_repeated_evaluations()
    print("All problem tests passed!")
