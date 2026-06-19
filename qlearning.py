import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

n_estados = 10
limites_bajos = env.observation_space.low
limites_altos = env.observation_space.high

limites_altos[1] = 3
limites_altos[3] = 3
limites_bajos[1] = -3
limites_bajos[3] = -3

tabla_q = np.zeros([n_estados] * 4 + [env.action_space.n])

import os
if os.path.exists("tabla_q.npy"):
    tabla_q = np.load("tabla_q.npy")
    print("Tabla cargada, continuando entrenamiento")
else:
    print("Tabla nueva, empezando desde cero")

def discretizar(observacion):
    indices = []
    for i in range(4):
        indice = int(np.digitize(observacion[i],
                     np.linspace(limites_bajos[i], limites_altos[i], n_estados)))
        indice = min(n_estados - 1, max(0, indice))
        indices.append(indice)
    return tuple(indices)

alpha = 0.1    # velocidad de aprendizaje
gamma = 0.99   # importancia del futuro
epsilon = 1.0  # exploración (empieza explorando todo)

for episodio in range(2000):
    obs, info = env.reset()
    estado = discretizar(obs)
    recompensa_total = 0

    for paso in range(200):
        if np.random.random() < epsilon:
            accion = env.action_space.sample()  # explorar
        else:
            accion = np.argmax(tabla_q[estado])  # explotar

        obs, recompensa, terminado, truncado, info = env.step(accion)
        nuevo_estado = discretizar(obs)

        tabla_q[estado][accion] = tabla_q[estado][accion] + alpha * (
            recompensa + gamma * np.max(tabla_q[nuevo_estado]) - tabla_q[estado][accion]
        )

        estado = nuevo_estado
        recompensa_total += recompensa

        if terminado or truncado:
            break

    epsilon = max(0.01, epsilon * 0.995)

    if episodio % 50 == 0:
        print(f"Episodio {episodio} | Recompensa: {recompensa_total:.0f} | Epsilon: {epsilon:.2f}")

np.save("tabla_q.npy", tabla_q)
print("Tabla guardada")

env.close()