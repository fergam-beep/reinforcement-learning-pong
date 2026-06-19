import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque
import os

gym.register_envs(ale_py)

def procesar_frame(obs):
    gris = np.mean(obs, axis=2).astype(np.uint8)
    recortado = gris[34:194, :]
    pequeño = recortado[::2, ::2]
    return pequeño.astype(np.float32) / 255.0

class DQN_Pong(nn.Module):
    def __init__(self, n_acciones):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, n_acciones)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacidad):
        self.buffer = deque(maxlen=capacidad)

    def guardar(self, estado, accion, recompensa, siguiente_estado, terminado):
        self.buffer.append((estado, accion, recompensa, siguiente_estado, terminado))

    def muestrear(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

env = gym.make("ALE/Pong-v5", render_mode=None)
n_acciones = env.action_space.n

red = DQN_Pong(n_acciones)
red_objetivo = DQN_Pong(n_acciones)

if os.path.exists("dqn_pong.pth"):
    red.load_state_dict(torch.load("dqn_pong.pth"))
    red_objetivo.load_state_dict(torch.load("dqn_pong.pth"))
    print("Modelo cargado, continuando entrenamiento")
else:
    red_objetivo.load_state_dict(red.state_dict())
    print("Modelo nuevo, empezando desde cero")

optimizador = torch.optim.Adam(red.parameters(), lr=0.0001)
buffer = ReplayBuffer(50000)

epsilon = float(np.load("epsilon.npy")) if os.path.exists("epsilon.npy") else 1.0
gamma = 0.99
batch_size = 32
pasos_totales = 0

print(f"Epsilon inicial: {epsilon:.2f}")

def preparar_estado(frame):
    estado = np.stack([frame] * 4, axis=0)
    return torch.tensor(estado).unsqueeze(0)

for episodio in range(3000):
    recompensa_total = 0
    obs, info = env.reset()
    frame = procesar_frame(obs)
    estado = preparar_estado(frame)

    for paso in range(10000):
        if random.random() < epsilon:
            accion = env.action_space.sample()
        else:
            with torch.no_grad():
                accion = red(estado).argmax().item()

        obs, recompensa, terminado, truncado, info = env.step(accion)
        siguiente_frame = procesar_frame(obs)
        siguiente_estado = preparar_estado(siguiente_frame)

        buffer.guardar(estado, accion, recompensa, siguiente_estado, terminado)
        estado = siguiente_estado
        recompensa_total += recompensa
        pasos_totales += 1

        if len(buffer) >= batch_size:
            experiencias = buffer.muestrear(batch_size)
            estados, acciones, recompensas, siguientes_estados, terminados = zip(*experiencias)

            estados = torch.cat(estados)
            acciones = torch.tensor(acciones)
            recompensas = torch.tensor(recompensas, dtype=torch.float32)
            siguientes_estados = torch.cat(siguientes_estados)
            terminados = torch.tensor(terminados, dtype=torch.float32)

            q_valores = red(estados).gather(1, acciones.unsqueeze(1)).squeeze()
            with torch.no_grad():
                q_objetivo = recompensas + gamma * red_objetivo(siguientes_estados).max(1).values * (1 - terminados)

            perdida = nn.MSELoss()(q_valores, q_objetivo)
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()

        if pasos_totales % 1000 == 0:
            red_objetivo.load_state_dict(red.state_dict())

        if terminado or truncado:
            break

    epsilon = max(0.01, epsilon * 0.995)

    if episodio % 10 == 0:
        print(f"Episodio {episodio} | Recompensa: {recompensa_total:.0f} | Epsilon: {epsilon:.2f}")

    if episodio % 50 == 0:
        torch.save(red.state_dict(), "dqn_pong.pth")
        np.save("epsilon.npy", epsilon)
        print(f"--- Checkpoint guardado ---")

torch.save(red.state_dict(), "dqn_pong.pth")
np.save("epsilon.npy", epsilon)
print("Entrenamiento completado")
env.close()