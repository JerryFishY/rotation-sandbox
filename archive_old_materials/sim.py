import numpy as np

def solve_cubic(a, b, c, d):
    return np.roots([a, b, c, d])

def compute_kn(en):
    a = 1 - en**3
    b = 6*en**2 - 6
    c = 9 - 9*en
    d = 0.0
    roots = solve_cubic(a, b, c, d)
    valid_roots = []
    for root in roots:
        if abs(root.imag) < 1e-10 and root.real > 1.0 + 1e-6:
            valid_roots.append(root.real)
    if not valid_roots:
        return 3.0
    return min(valid_roots)

en = 1.0 / 4096.0
alpha = 0.001
iteration = 0

print(f"Target: reach en >= {1.0-alpha}")
while en < (1.0 - alpha) and iteration < 15:
    kn = compute_kn(en)
    en_next = kn * en * ((3.0 - kn * en) ** 2) / 4.0
    print(f"Iter {iteration}: en={en:.6f} -> {en_next:.6f}, kn={kn:.6f}")
    en = en_next
    iteration += 1

print(f"Total iterations: {iteration}")
