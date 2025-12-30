import numpy as np
import DNLP_diff_engine as dnlp

x = dnlp.make_variable(3, 1, 0, 3)
log_x = dnlp.make_log(x)

u = np.array([1.0, 2.0, 3.0])
out = dnlp.forward(log_x, u)
print("log(x) forward:", out)
