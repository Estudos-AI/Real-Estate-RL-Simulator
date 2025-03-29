import pygame
import time
import sys
import os
import importlib
import random

# Caminhos
sys.path.append(os.path.abspath("."))

# Imports do ambiente e mapa
import RL.environments.HomeChoice_v0 as home_env
importlib.reload(home_env)
HomeChoiceEnv = home_env.HomeChoiceEnv

from RL.environments.GEO.maps.SP import distritos

# Seed para consistência na posição dos ícones
random.seed(42)

# Cores
COLOR_BG = (240, 240, 240)
COLOR_DISTRICT = (0, 0, 0)
COLOR_OUTLINE = (0, 0, 0)
COLOR_AVAILABLE = (100, 149, 237)   # Azul
COLOR_ANALYZED = (255, 215, 0)      # Amarelo
COLOR_BOUGHT = (60, 179, 113)       # Verde
COLOR_SOLD = (220, 20, 60)          # Vermelho

ICON_RADIUS = 5


def draw_distritos(screen):
    for poly in distritos:
        pygame.draw.polygon(screen, COLOR_DISTRICT, poly, width=0)
        pygame.draw.polygon(screen, COLOR_OUTLINE, poly, width=1)


def draw_imoveis(screen, mercado, current_step, comprados, vendidos):
    for i, prop in enumerate(mercado[current_step:current_step+20]):
        if "pos" not in prop:
            # Posição aleatória temporária (depois pode ser vinculada ao bairro)
            x = random.randint(100, 700)
            y = random.randint(100, 500)
            prop["pos"] = (x, y)

        color = COLOR_ANALYZED if i == 0 else COLOR_AVAILABLE
        pygame.draw.circle(screen, color, prop["pos"], ICON_RADIUS)

    for prop in comprados:
        if "pos" in prop:
            pygame.draw.circle(screen, COLOR_BOUGHT, prop["pos"], ICON_RADIUS)

    for prop in vendidos:
        if "pos" in prop:
            pygame.draw.circle(screen, COLOR_SOLD, prop["pos"], ICON_RADIUS)


def draw_hud(screen, env, step):
    font = pygame.font.SysFont("Arial", 18)
    patrimonio = env.cash + env._calculate_property_value()
    hud = [
        f"Passo: {step}",
        f"Saldo: R${env.cash:,.0f}",
        f"Imóveis: {len(env.owned_properties)}",
        f"Patrimônio: R${patrimonio:,.0f}"
    ]
    for i, text in enumerate(hud):
        rendered = font.render(text, True, (0, 0, 0))
        screen.blit(rendered, (10, 10 + i * 22))


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("🏡 Real Estate RL - Mapa de SP")
    clock = pygame.time.Clock()

    env = HomeChoiceEnv()
    obs = env.reset()
    vendidos = []

    for step in range(500):
        screen.fill(COLOR_BG)

        # Atualiza ambiente
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)

        # Armazena venda se houver
        if action == 2 and hasattr(env, "last_sold"):
            vendidos.append(env.last_sold)
            delattr(env, "last_sold")

        # Desenhos
        draw_distritos(screen)
        draw_imoveis(screen, env.market, env.current_step, env.owned_properties, vendidos)
        draw_hud(screen, env, step)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                exit()

        time.sleep(0.1)
        clock.tick(60)

        if done:
            break

    pygame.quit()
