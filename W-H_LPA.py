import numpy as np
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Il codice in questo momento risolve in LPA la W-H, restituendo la massa rinormalizzata e il potenziale finale ricostruito,
# devo adesso implementare la ricerca del minimo del potenziale per poter approcciare anche casi asimmetrici




# Inizio del timer
start_time = time.time() #questo è solamente un timer che uso per avere una stima del tempo richiesto al codice

# Define domain and parameters, qui definisco semplicemente i parametri iniziali
L = 7
N = 1001  # Number of spatial points
dx = 2 * L / N
x = np.linspace(-L, L, N)
a2 = -0.5
a4 = 0.1/3.14
tin= -7
tfin=10


# Initial conditions, qui invece do le condizioni iniziali di quello che andiamo a risolvere
vfinal0 = a2 * x**2 + a4 * x**4  # Condizione iniziale del potenziale
v_second = 2 * a2 + 12 * a4 * x**2 # Derivata seconda del potenziale
initial_w = np.log(1 + v_second * np.exp(2 * tin))  # Condizione iniziale di w(x,0)
w0 = np.full(N, initial_w) # Qui creo un array di dimensione N che contiene la condizione iniziale di w
initial_conditions = np.hstack((w0, vfinal0)) # Qui metto insieme le due condizioni iniziali in un unico array per il modello da risolvere





# Define the model function for the solver, qui dentro si possono modificare le equazioni da risolvere

def model(t, y):
    N = len(y) // 2 # prendo come dimensione delle soluzioni
    w = y[:N].copy()  # Questo e il seguente sono i vettori che contengono le soluzioni
    vfinal = y[N:]
    dwdt = np.zeros_like(w) #Questo e il seguente sono i vettori che contengono le equazioni differenziali
    dvfinaldt = np.zeros_like(vfinal)
    
    # Dynamic boundary condition update, calcolo le condizioni al contorno ad ogni istante temporale
    v_second1 = 2 * a2 + 12 * a4 * L**2
    boundary_value = np.log(1 + v_second1 * np.exp(2 * t))


    # Apply boundary conditions to w
    w[0] = w[-1] = boundary_value #Qui calcolo le boundary prima di definire l'equazione così da fare la giusta discretizzazione
    

    wxx = (np.roll(w, -1) - 2 * w + np.roll(w, 1)) / dx**2 #Questa è la derivata seconda discretizzata con le differenze centrali
 
    # Calculate the derivative based on the PDE
    dwdt[1:-1] = (np.exp(t - w[1:-1]) * wxx[1:-1]) / (2 ) - 2 * np.exp(-w[1:-1]) + 2 #Questa è l'equazione per w da risolvere

    # Update for Vfinal
    dvfinaldt = 0.5 * np.exp(-t) * w #Questa è l'equazione per il potenziale che sto accoppiando per riscotruire il potenziale durante l'integrazione

    

    y[:N] = w

    return np.hstack((dwdt, dvfinaldt))

# Time span for the solver, indico gli estremi temporali d'integrazioni e quanti punti prendere nel tempo
t_span = [tin, tfin]
t_eval = np.linspace(t_span[0], t_span[1], 200)

# Solve the system of ODEs, qui risolvo le equazioni, il tempo impiegato è circa di 20 secondi, si può diminuire un po' diminuendo la tolleranza richiesta
sol = solve_ivp(model, t_span, initial_conditions, t_eval=t_eval, method='LSODA', atol=1e-6, rtol=1e-6)




#Post simulation, qui faccio un paio di sistemazioni alle soluzioni. Principalmente su w, siccome nella funzione che rappresenta le nostre equazioni
#le condizioni al contorno sono calcolate prima della risoluzione dell'equazione, vengono sovrascritte dopo l'ultima integrazione e quindi devono essere aggiunte dopo.
#Questo problema non si ha per il potenziale


w_final = sol.y[:N, -1]
v_second1 = 2 * a2 + 12 * a4 * L**2
final_boundary=np.log(1 + v_second1 * np.exp(2 * tfin))
w_final[0]=final_boundary
w_final[-1]=final_boundary
v_final_final = sol.y[N:, -1]
# Calcolo la derivata seconda invertendo la relazione con w(x,t)
V_second = np.exp(-2 * tfin) * (np.exp(w_final) - 1)
# Index corresponding to x=0
index_x0 = N // 2
# Compute V''(x) at x=0
V_second_x0 = np.exp(-2 * tfin) * (np.exp(w_final[index_x0]) - 1)
sqrt_V_second_x0 = np.sqrt(V_second_x0)
print("Square root of V''(x) at x=0:", sqrt_V_second_x0)
# Fine del timer
end_time = time.time()
# Calcolo e stampa del tempo di esecuzione
tempo_esecuzione = end_time - start_time
print(f"Tempo di esecuzione: {tempo_esecuzione} secondi")


# Plot della derivata seconda al tempo finale
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(x, V_second, label="V''(x)")
plt.title('Plot of V''(x) vs x')
plt.xlabel('x')
plt.ylabel('v_final')
plt.legend()
plt.grid(True)

# Plot del potenziale al tempo finale
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(x, v_final_final, label='v_final at t_final')
plt.title('Final v_final vs x')
plt.xlabel('x')
plt.ylabel('v_final')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()