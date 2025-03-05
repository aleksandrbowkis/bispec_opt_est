import sympy as sp
from sympy.physics.quantum.cg import Wigner3j

# Calculate a Wigner 3j symbol
# Format: Wigner3j(j1, j2, j3, m1, m2, m3)
result = Wigner3j(1, 1, 2, 0, 0, 0)
print(result)
# Evaluate to a numerical value if needed
numerical_result = float(result.doit())
print(numerical_result)

numerical_result = float(Wigner3j(1, 1, 2, 0, 0, 0).doit())