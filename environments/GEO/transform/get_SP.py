
# SÃO PAULO - MAPA DE DISTRITOS
# Este script gera um mapa de contorno dos distritos da cidade de São Paulo, a partir de um arquivo GeoJSON.
###################################################################################################################
# import pandas as pd
# import plotly.express as px
# import json

# # Caminho local do seu GeoJSON
# geojson_path = "RL/environments/distritos.geojson"

# # Carrega o arquivo
# with open(geojson_path, encoding="utf-8") as f:
#     geojson = json.load(f)

# # Extrai nomes dos distritos do campo 'ds_nome'
# nomes_distritos = [feature['properties']['ds_nome'] for feature in geojson['features']]

# # Cria dados fictícios de valor médio dos imóveis
# df = pd.DataFrame({
#     "ds_nome": nomes_distritos,
#     "dummy": [1] * len(nomes_distritos)  # valor fixo
# })

# # Cria o mapa com Plotly
# fig = px.choropleth_mapbox(
#     df,
#     geojson=geojson,
#     locations="ds_nome",
#     featureidkey="properties.ds_nome",
#     color="dummy",  # necessário, mas é um valor fixo
#     color_continuous_scale=[[0, "white"], [1, "lightgrey"]],
#     mapbox_style="carto-positron",
#     zoom=9.8,
#     center={"lat": -23.55, "lon": -46.63},
#     opacity=1
# )

# fig.update_traces(showscale=False)
# fig.update_layout(
#     title="Distritos de São Paulo (apenas contorno)",
#     margin={"r": 0, "t": 40, "l": 0, "b": 0}
# )
# fig.show()
###################################################################################################################

import geopandas as gpd
import os

# Caminho do GeoJSON original
geojson_path = os.path.join("RL", "environments", "distritos.geojson")

# Caminho de saída para o arquivo com contornos convertidos
output_path = os.path.join("GEO", "maps", "sp_outline.py")

# Carrega os dados geográficos
gdf = gpd.read_file(geojson_path)

# Projeta para sistema de coordenadas em metros (Web Mercator)
gdf = gdf.to_crs(epsg=3857)

# Converte geometria para coordenadas pygame-friendly
def geo_to_pygame_coords(geometry, scale=0.0001, offset=(400, 300)):
    if geometry.geom_type == "Polygon":
        return [[(x * scale + offset[0], -y * scale + offset[1]) for x, y in geometry.exterior.coords]]
    elif geometry.geom_type == "MultiPolygon":
        return [
            [(x * scale + offset[0], -y * scale + offset[1]) for x, y in poly.exterior.coords]
            for poly in geometry.geoms
        ]
    else:
        return []

# Converte todos os distritos
pygame_polygons = []
for geom in gdf.geometry:
    polygons = geo_to_pygame_coords(geom)
    pygame_polygons.extend(polygons)

# Gera arquivo .py com os contornos
with open(output_path, "w", encoding="utf-8") as f:
    f.write("# Contornos dos distritos de São Paulo em coordenadas pygame\n")
    f.write("distritos = [\n")
    for poly in pygame_polygons:
        f.write("    " + str(poly) + ",\n")
    f.write("]\n")

print(f"✅ Contornos salvos em '{output_path}' com {len(pygame_polygons)} polígonos.")


