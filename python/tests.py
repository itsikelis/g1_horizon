import casadi as cs

x = cs.MX.sym("x")

y = cs.SX.sym("y", 5)

Z = cs.SX.sym("Z", 5, 5)

f = x**2 + 10
f = cs.sqrt(f)

print(f)
