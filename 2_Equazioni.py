import numpy as np
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Inizio del timer
start_time = time.time()
# Define domain and parameters
L = 7
N = 1000  # Number of spatial points
dx = 2 * L / N
x = np.linspace(-L, L, N)

# Initial conditions
u0 = np.exp(-x**2)
v0 = np.exp(-x**2)
initial_conditions = np.hstack((u0, v0))

# Function to calculate derivatives
def model(t, y):
    u = y[:N]
    v = y[N:]
    dudt = np.zeros_like(u)
    dvdt = np.zeros_like(v)

    # Apply central difference formula for spatial derivatives
    dudx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    dvdx2 = (np.roll(v, -1) - 2 * v + np.roll(v, 1)) / dx**2
    dudx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    dvdx = (np.roll(v, -1) - np.roll(v, 1)) / (2 * dx)
     # Including the squared derivative of v and linear derivative of u in dvdt calculation
    dvdx_squared = dvdx**2  # Square of the first derivative of v

    # Correctly including the coupling terms in the time derivatives
    # Rearrange the equation to:
    # dudt = dudx2 - dvdt  --> As dudt = dudx2 - dvdt, we can't solve it directly without iteration or further algebraic manipulation
    # dvdt = dvdx2 - dudx - dvdx  --> This we can calculate directly
    dvdt[1:-1] = dvdx2[1:-1] - dvdx_squared[1:-1] - dudx[1:-1]
    
    # We need to approximate or iteratively solve for dudt since it is dependent on dvdt
    # For a first pass, let's ignore dvdt's time dependency in the calculation of dudt and see if further adjustments are needed:
    dudt[1:-1] = dudx2[1:-1] - dvdt[1:-1]  # Assuming immediate coupling effect, this is a simplification

    # Boundary conditions
    dudt[0], dudt[-1] = 0, 0
    dvdt[0], dvdt[-1] = 0, 0

    return np.hstack((dudt, dvdt))




# Define the time span for the solver
t_span = [0, 5]
t_eval = np.linspace(t_span[0], t_span[1], 200)

# Solve the system of ODEs using the BDF method for stiff problems
sol = solve_ivp(model, t_span, initial_conditions, t_eval=t_eval, method='BDF', atol=1e-10, rtol=1e-6)


# Fine del timer
end_time = time.time()
# Calcolo e stampa del tempo di esecuzione
tempo_esecuzione = end_time - start_time
print(f"Tempo di esecuzione: {tempo_esecuzione} secondi")

# Plot the results for u and v at the final time
plt.figure(figsize=(12, 6))
plt.plot(x, sol.y[:N, -1], label='u(x, T)')
plt.plot(x, sol.y[N:, -1], label='v(x, T)')
plt.title('Solution at T = 5 using Method of Lines')
plt.xlabel('x')
plt.ylabel('u, v')
plt.legend()
plt.show()
# Fine del timer
end_time = time.time()
# Calcolo e stampa del tempo di esecuzione
tempo_esecuzione = end_time - start_time
print(f"Tempo di esecuzione: {tempo_esecuzione} secondi")