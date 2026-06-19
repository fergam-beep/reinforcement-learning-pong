import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

# ── Red neuronal ──────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, n_observaciones, n_acciones):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(n_observaciones, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_acciones)
        )

    def forward(self, x):
        return self.red(x)

# ── Replay Buffer ─────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacidad):
        self.buffer = deque(maxlen=capacidad)

    def guardar(self, estado, accion, recompensa, siguiente_estado, terminado):
        self.buffer.append((estado, accion, recompensa, siguiente_estado, terminado))

    def muestrear(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ── Configuración ─────────────────────────────────────────
env = gym.make("CartPole-v1", render_mode="human")

n_observaciones = env.observation_space.shape[0]
n_acciones = env.action_space.n

red = DQN(n_observaciones, n_acciones)
red_objetivo = DQN(n_observaciones, n_acciones)
red_objetivo.load_state_dict(red.state_dict())

optimizador = torch.optim.Adam(red.parameters(), lr=0.001)
buffer = ReplayBuffer(10000)

epsilon = 1.0
gamma = 0.99
batch_size = 64
actualizar_objetivo_cada = 100
pasos_totales = 0

# ── Entrenamiento ─────────────────────────────────────────
for episodio in range(500):
    obs, info = env.reset()
    estado = torch.tensor(obs, dtype=torch.float32)
    recompensa_total = 0

    for paso in range(500):
        # Elegir acción
        if random.random() < epsilon:
            accion = env.action_space.sample()
        else:
            with torch.no_grad():
                accion = red(estado).argmax().item()

        # Ejecutar acción
        obs, recompensa, terminado, truncado, info = env.step(accion)
        siguiente_estado = torch.tensor(obs, dtype=torch.float32)

        buffer.guardar(estado, accion, recompensa, siguiente_estado, terminado)
        estado = siguiente_estado
        recompensa_total += recompensa
        pasos_totales += 1

        # Aprender
        if len(buffer) >= batch_size:
            experiencias = buffer.muestrear(batch_size)
            estados, acciones, recompensas, siguientes_estados, terminados = zip(*experiencias)

            estados = torch.stack(estados)
            acciones = torch.tensor(acciones)
            recompensas = torch.tensor(recompensas, dtype=torch.float32)
            siguientes_estados = torch.stack(siguientes_estados)
            terminados = torch.tensor(terminados, dtype=torch.float32)

            q_valores = red(estados).gather(1, acciones.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_objetivo = recompensas + gamma * red_objetivo(siguientes_estados).max(1).values * (1 - terminados)

            perdida = nn.MSELoss()(q_valores, q_objetivo)
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()

        # Actualizar red objetivo
        if pasos_totales % actualizar_objetivo_cada == 0:
            red_objetivo.load_state_dict(red.state_dict())

        if terminado or truncado:
            break

    epsilon = max(0.01, epsilon * 0.995)

    if episodio % 10 == 0:
        print(f"Episodio {episodio} | Recompensa: {recompensa_total:.0f} | Epsilon: {epsilon:.2f}")

torch.save(red.state_dict(), "dqn_cartpole.pth")
print("Modelo guardado")
env.close()