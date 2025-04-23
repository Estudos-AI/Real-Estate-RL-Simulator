import sys
import os
import importlib
import matplotlib.pyplot as plt
import numpy as np

# ===== PARÂMETROS PERSONALIZÁVEIS =====
NUM_IMOVEIS   = 10000
NUM_EPISODIOS = 500
NUM_STEPS     = 240
# =====================================

sys.path.append(os.path.abspath("."))

import environments.HomeChoice_v2 as home_env
importlib.reload(home_env)
HomeChoiceEnv = home_env.HomeChoiceEnv

# Instancia o ambiente fixo com mercado gerado uma única vez
env = HomeChoiceEnv(max_steps=NUM_STEPS, num_imoveis=NUM_IMOVEIS, verbose=False)
env.market = env._generate_market()

# Coleta de resultados
episodios_recompensa_total = []
episodios_patrimonio_final = []
episodios_steps_usados = []
episodios_historico = []

# Loop de simulação
for ep in range(NUM_EPISODIOS):
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = env.action_space.sample()  # Substitua por sua policy futuramente
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    episodios_recompensa_total.append(total_reward)
    episodios_patrimonio_final.append(env._get_total_wealth())
    episodios_steps_usados.append(env.current_step)
    episodios_historico.append(env.history.copy())

    print(f"\n Episódio {ep + 1}/{NUM_EPISODIOS} | Recompensa total: {total_reward:.2f} | "
          f"Patrimônio final: R${env._get_total_wealth():,.2f} | Steps: {env.current_step}")

# =======================
# Curvas de aprendizado
# =======================
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Recompensa por episódio
axs[0].plot(episodios_recompensa_total, label='Recompensa Total', color='blue')
axs[0].set_title("Recompensa por Episódio")
axs[0].set_xlabel("Episódio")
axs[0].set_ylabel("Recompensa")
axs[0].grid(True)

# Patrimônio por episódio
axs[1].plot(episodios_patrimonio_final, label='Patrimônio Final', color='green')
axs[1].set_title("Patrimônio Final por Episódio")
axs[1].set_xlabel("Episódio")
axs[1].set_ylabel("R$")
axs[1].grid(True)

plt.tight_layout()
plt.show()


# =======================
# Animação por Episódio
# =======================
env.render_timelapse_ep(episodios_historico)
