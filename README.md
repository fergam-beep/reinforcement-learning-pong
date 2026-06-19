# Reinforcement Learning Pong Agent

Este es mi primer proyecto serio de Reinforcement Learning. La idea es sencilla: entrenar un agente que aprenda a jugar Pong sin que yo le diga las reglas, solo dejándolo jugar miles de partidas hasta que descubra por sí mismo cómo ganar.

Empecé sin tener ni idea de RL, así que fui subiendo de nivel poco a poco en lugar de tirarme directo a Pong.

## El camino hasta aquí

Primero necesitaba entender el concepto básico: un agente observa, actúa, recibe una recompensa, y repite. Para eso usé CartPole (el clásico palo que hay que equilibrar), que es mucho más simple que Pong.

1. Empecé con Q-Learning tabular, básicamente una tabla gigante donde el agente apunta qué tan buena es cada acción en cada situación.
2. Luego pasé a DQN (Deep Q-Network), en lugar de una tabla, una red neuronal que aprende a predecir qué acción es mejor.
3. Con eso ya entendido, salté a Pong, que es mucho más complicado porque el agente no ve 4 números como en CartPole, ve una imagen entera de píxeles.

## Cómo va el progreso

- CartPole con Q-Learning tabular: funciona, aunque se queda corto (alrededor de 25-30 de recompensa de un máximo razonable)
- CartPole con DQN: esto sí funciona de verdad, llega al máximo de 500 de forma consistente
- Pong con DQN: entrenando ahora mismo. Pong es mucho más lento de aprender que CartPole, así que esto va a tardar días

## Cómo funciona por dentro

Las imágenes de Pong se simplifican antes de meterlas en la red: se pasan a escala de grises, se recortan los bordes inútiles y se reducen de tamaño.

Una red neuronal convolucional (CNN) mira esas imágenes y decide qué acción tomar.

El agente guarda sus partidas en una especie de cuaderno de experiencias (replay buffer) y aprende repasando recuerdos al azar, no solo lo último que le pasó.

Hay dos copias de la red, una que aprende todo el rato y otra que se queda quieta un poco más, para que el entrenamiento no sea tan caótico.

Al principio el agente actúa totalmente al azar (exploración) y poco a poco va confiando más en lo que ha aprendido.

## Estructura del proyecto

qlearning.py: Q-Learning con tabla, en CartPole
dqn.py: DQN en CartPole
dqn_pong.py: DQN entrenando en Pong, el proyecto principal
pong.py: agente jugando al azar, para comparar
ver_pong.py: para ver jugar al agente ya entrenado

## Resultados

Todavía entrenando. En cuanto tenga algo decente voy a subir una gráfica de cómo mejora la recompensa con el tiempo, y un vídeo del agente jugando.

## Lo que he ido aprendiendo

El loop básico de cualquier problema de RL: observar, actuar, recibir recompensa, repetir.

Por qué hace falta balancear exploración y explotación, y qué pasa si no lo haces bien.

Por qué entrenar una red contra sí misma es inestable, y para qué sirve la red objetivo.

Que cuando el problema se vuelve más complejo, de 4 números a una imagen entera, hace falta una red distinta (CNN).

Que esto de entrenar agentes lleva tiempo de verdad, no es como entrenar un modelo en cinco minutos.

## Por hacer

Dejar entrenar lo suficiente para ver mejora clara.

Gráfica de recompensa por episodio.

Grabar un vídeo del agente jugando ya entrenado.

Probar a ajustar hiperparámetros si no mejora lo suficiente.
