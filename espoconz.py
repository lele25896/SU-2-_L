import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
L = 7
N = 200
dx = 2 * L / N
x = np.linspace(-L, L, N)
t_in = -np.log(1000)
t_fin=5
a2 = -0.5  # Example value for a2, adjust as needed
a4=0.1


# Initial conditions
v0 = a2 * x**2+a4*x**4
z0 = np.ones_like(x)
initial_conditions = np.hstack((v0, z0))

# Model definition
def model(t, y):
    v = np.copy(y[:N])
    z = np.copy(y[N:])
    
    # Apply boundary conditions
    v[0], v[-1] = a2 * L**2+a4*L**4, a2 * L**2+a4*L**4  # v(t, ±L) = a2 * L^2
    z[0], z[-1] = 1, 1                 # z(t, ±L) = 1

    dvdt = np.zeros_like(v)
    dzdt = np.zeros_like(z)

    v_xx = (np.roll(v, -1) - 2 * v + np.roll(v, 1)) / dx**2
    v_xxx = (-np.roll(v, -2) + 2*np.roll(v, -1) - 2*np.roll(v, 1) + np.roll(v, 2)) / (2 * dx**3)
    z_xx = (np.roll(z, -1) - 2 * z + np.roll(z, 1)) / dx**2
    z_x = (np.roll(z, -1) - np.roll(z, 1)) / (2 * dx)

    exp_term_v = np.exp(-v_xx / (np.exp(-2 * t) * z))
    exp_term_v = np.clip(exp_term_v, -100, 100)  # Clip values to prevent overflow
    complex_term = (-z_xx / (z * np.exp(-2 * t)) + 
                    7 * z_x**2 / (8 * z**2 * np.exp(-2 * t)) +
                    3 * z_x * v_xx / (2 * (z * np.exp(-2 * t))**2) -
                    z * v_xxx**2 / (6 * (z * np.exp(-2 * t))**3))

    dvdt[1:-1] = -np.exp(-t) / np.sqrt(4 * np.pi) * exp_term_v[1:-1]
    dzdt[1:-1] = -np.exp(-t) / np.sqrt(4 * np.pi) * exp_term_v[1:-1] * complex_term[1:-1]

    return np.hstack((dvdt, dzdt))

# Solve the system of ODEs using an appropriate method
t_span = [t_in, t_fin]
t_eval = np.linspace(t_span[0], t_span[1], 200)
sol = solve_ivp(model, t_span, initial_conditions, t_eval=t_eval, method='BDF')

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x, sol.y[:N, -1], label='v(x, T)')
plt.plot(x, sol.y[N:, -1], label='z(x, T)')
plt.title(f"Solution at T = {t_fin} using Method of Lines")
plt.xlabel('x')
plt.ylabel('v, z')
plt.legend()
plt.show()

# Calculate the square root of v_xx(x=0)/z(x=0) at final time
central_index = N // 2
v_xx_final = (sol.y[central_index+1, -1] - 2 * sol.y[central_index, -1] + sol.y[central_index-1, -1]) / dx**2
z_final = sol.y[N + central_index, -1]
result = np.sqrt(v_xx_final / z_final)

print(f"Square root of v_xx(x=0)/z(x=0) at final time: {result}")
print({z_final})
