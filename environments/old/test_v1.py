import sys
import os
import importlib

# ===== PARÂMETROS PERSONALIZÁVEIS =====
NUM_IMOVEIS = 10000       # 🔧 Quantos imóveis serão gerados
NUM_EPISODIOS = 500       # 🔁 Quantos passos de simulação
SAVE_PATH = "environments/GEO/tests/mapa_animado_v1.html"  # 📁 Caminho de saída do HTML
# =====================================

sys.path.append(os.path.abspath("."))

import environments.HomeChoice_v2 as home_env
importlib.reload(home_env)
HomeChoiceEnv = home_env.HomeChoiceEnv

# Instancia ambiente com número de imóveis definido
env = HomeChoiceEnv()
env.num_imoveis = NUM_IMOVEIS
env.market = env._generate_market()
obs = env.reset()
env.vendidos = []

# Guarda histórico dos imóveis por ID
historico_imoveis = {}

for step in range(NUM_EPISODIOS):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

    # Garante que estamos dentro do range de mercado
    if env.current_step < len(env.market):
        prop = env.market[env.current_step]
        prop_id = prop["id"]

        if prop_id not in historico_imoveis:
            historico_imoveis[prop_id] = {
                "prop": prop,
                "step": 0,          # quando ele apareceu na tela
                "owned": False,
                "sold": False
            }

        # Se comprou, marca como "owned"
        if action == 0:
            historico_imoveis[prop_id]["owned"] = True
            historico_imoveis[prop_id]["step"] = step

    # Se vendeu, marca como "sold"
    if action == 2 and hasattr(env, "last_sold"):
        prop = env.last_sold
        prop_id = prop["id"]
        if prop_id in historico_imoveis:
            historico_imoveis[prop_id]["sold"] = True
            historico_imoveis[prop_id]["step"] = step
        env.vendidos.append(prop)
        delattr(env, "last_sold")

    if done:
        break

# Converte dicionário em lista para renderização
historico_list = list(historico_imoveis.values())

# Renderiza visualização animada
env.render_folium_timelapse_v1(historico_list, save_path=SAVE_PATH)
