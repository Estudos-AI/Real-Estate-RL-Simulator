#import pygame
import time
import sys
import os
import importlib
import folium
import geemap.foliumap as geemap

sys.path.append(os.path.abspath("."))

import environments.HomeChoice_v1 as home_env
importlib.reload(home_env)
HomeChoiceEnv = home_env.HomeChoiceEnv

env = HomeChoiceEnv()
obs = env.reset()
env.vendidos = []

for step in range(500):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

    if action == 2 and hasattr(env, "last_sold"):
        env.vendidos.append(env.last_sold)
        delattr(env, "last_sold")

    env.render_geemap_folium_v1()