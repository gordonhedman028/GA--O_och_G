import matplotlib.pyplot as plt
import numpy as np


m = 10       # Massa
c = 0.1      # Dämpningskonstant
k = 1        # Fjäderkonstant

# Initialvillkor
p0 = 1
a0 = 0
v0 = 0

# Tid
dt = 1.5
t_max = 100
steps = int(t_max / dt)

#Eulers stegmetod

p = p0
v = v0

pos1 = [p0]

for i in range(steps):
    a = -(k*p + c*v)/m

    v += a*dt
    p += v*dt

    pos1.append(p)


#RK4

p = p0
v = v0

pos2 = [p0]

def acceleration(p, v):
    return -(k*p + c*v)/m

for i in range(steps):

    # k1
    k1p = v
    k1v = acceleration(p, v)

    # k2
    k2p = v + dt*k1v/2
    k2v = acceleration(
        p + dt*k1p/2,
        v + dt*k1v/2
    )

    # k3
    k3p = v + dt*k2v/2
    k3v = acceleration(
        p + dt*k2p/2,
        v + dt*k2v/2
    )

    # k4
    k4p = v + dt*k3v
    k4v = acceleration(
        p + dt*k3p,
        v + dt*k3v
    )

    # uppdatera
    p += dt/6 * (k1p + 2*k2p + 2*k3p + k4p)
    v += dt/6 * (k1v + 2*k2v + 2*k3v + k4v)

    pos2.append(p)


#Lösning

omega = np.sqrt(k/m - c**2/(4*m**2))

x2 = np.arange(0, t_max + dt, 0.01)

y2 = np.exp(-c/(2*m)*x2) * (
    p0*np.cos(omega*x2)
    + (v0 + c*p0/(2*m))/omega * np.sin(omega*x2)
)


#Rita

x1 = np.linspace(0, t_max, steps + 1)

plt.figure(figsize=(10, 6))

plt.plot(x2, y2, label="Analytical solution")
plt.plot(x1, pos1, label="Euler")
plt.plot(x1, pos2, label="RK4")

plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Damped spring: Numerical vs analytical solution")

plt.legend()
plt.grid()

plt.show()
