import pygame
import time
import sys
import os
import importlib

# Caminhos
sys.path.append(os.path.abspath("."))

# Importa ambiente e mapa
import RL.environments.HomeChoice_v0 as home_env
importlib.reload(home_env)
HomeChoiceEnv = home_env.HomeChoiceEnv

import RL.environments.GEO.maps as mapitos
distritos = mapitos.distritos

# Cores para distritos e imóveis
COLOR_BG = (240, 240, 240)
COLOR_DISTRICT = (180, 180, 180)
COLOR_OUTLINE = (50, 50, 50)

COLOR_AVAILABLE = (100, 149, 237)   # Azul - disponível
COLOR_ANALYZED = (255, 215, 0)      # Amarelo - imóvel atual
COLOR_BOUGHT = (60, 179, 113)       # Verde - comprado
COLOR_SOLD = (220, 20, 60)          # Vermelho - vendido

ICON_RADIUS = 5  # Tamanho dos ícones no mapa

# Posição aleatória dos imóveis no mapa por enquanto
import random
random.seed(42)


def draw_distritos(screen):
    for poly in distritos:
        pygame.draw.polygon(screen, COLOR_DISTRICT, poly, width=0)
        pygame.draw.polygon(screen, COLOR_OUTLINE, poly, width=1)


def draw_imoveis(screen, mercado, current_step, comprados, vendidos):
    for i, prop in enumerate(mercado[current_step:current_step+20]):
        x = random.randint(100, 700)
        y = random.randint(100, 500)
        prop["pos"] = (x, y)  # Salva posição temporária

        color = COLOR_AVAILABLE
        if i == 0:
            color = COLOR_ANALYZED
        pygame.draw.circle(screen, color, (x, y), ICON_RADIUS)

    for prop in comprados:
        if "pos" in prop:
            pygame.draw.circle(screen, COLOR_BOUGHT, prop["pos"], ICON_RADIUS)

    for prop in vendidos:
        if "pos" in prop:
            pygame.draw.circle(screen, COLOR_SOLD, prop["pos"], ICON_RADIUS)


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("🏡 Real Estate RL - Mapa de SP")
    clock = pygame.time.Clock()

    env = HomeChoiceEnv()
    obs = env.reset()

    vendidos = []

    for _ in range(500):
        screen.fill(COLOR_BG)

        # Atualiza ambiente
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        # Detecta venda (ação 2) e salva no histórico
        if action == 2 and hasattr(env, "last_sold"):
            vendidos.append(env.last_sold)
            delattr(env, "last_sold")

        # Desenha mapa
        draw_distritos(screen)

        # Desenha imóveis com ícones diferentes e cores
        draw_imoveis(screen, env.market, env.current_step, env.owned_properties, vendidos)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        time.sleep(0.1)
        clock.tick(60)

        if done:
            break

    pygame.quit()
