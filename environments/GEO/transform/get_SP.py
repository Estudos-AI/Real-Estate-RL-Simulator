# SÃO PAULO - MAPA DE DISTRITOS
# Este script gera um mapa de contorno dos distritos da cidade de São Paulo, a partir de um arquivo GeoJSON.
# Permite visualizar o mapa antes de exportar para coordenadas pygame.

import geopandas as gpd
import matplotlib.pyplot as plt
import os

# Caminhos
geojson_path = os.path.join("RL", "environments", "GEO", "raw" , "distritos.geojson")
output_path  = os.path.join("RL", "environments", "GEO", "maps", "SP.py")

# Parâmetros ajustáveis
scale = 0.00005         # quanto menor, mais cabe na tela
offset = (400, 550)     # move o mapa no plano (x, y)

# Carrega o GeoJSON e reprojeta para metros
gdf = gpd.read_file(geojson_path)
gdf = gdf.to_crs(epsg=3857)


def geo_to_pygame_coords(geometry, scale, offset):
    if geometry.geom_type == "Polygon":
        return [[(x * scale + offset[0], -y * scale + offset[1]) for x, y in geometry.exterior.coords]]
    elif geometry.geom_type == "MultiPolygon":
        return [
            [(x * scale + offset[0], -y * scale + offset[1]) for x, y in poly.exterior.coords]
            for poly in geometry.geoms
        ]
    return []


# Converte todos os distritos
pygame_polygons = []
for geom in gdf.geometry:
    polygons = geo_to_pygame_coords(geom, scale=scale, offset=offset)
    pygame_polygons.extend(polygons)

# Visualização com matplotlib
plt.figure(figsize=(8, 6))
for poly in pygame_polygons:
    xs, ys = zip(*poly)
    plt.plot(xs, ys, color="black", linewidth=0.7)
plt.title("Pré-visualização do mapa vetorial (ajuste scale e offset se necessário)")
plt.gca().invert_yaxis()
plt.axis("equal")
plt.tight_layout()
plt.show()

# Exporta como arquivo .py com lista distritos
with open(output_path, "w", encoding="utf-8") as f:
    f.write("# Contornos dos distritos de São Paulo em coordenadas pygame\n")
    f.write("distritos = [\n")
    for poly in pygame_polygons:
        f.write("    " + str(poly) + ",\n")
    f.write("]\n")

print(f"✅ Contornos salvos em '{output_path}' com {len(pygame_polygons)} polígonos.")
print(f"📐 Scale usado: {scale}")
print(f"🧭 Offset usado: {offset}")
