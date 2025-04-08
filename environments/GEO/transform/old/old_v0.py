import pandas as pd
import plotly.express as px
import json

geojson_path = "RL/environments/distritos.geojson"
with open(geojson_path, encoding="utf-8") as f:
    geojson = json.load(f)
nomes_distritos = [feature['properties']['ds_nome'] for feature in geojson['features']]
df = pd.DataFrame({
    "ds_nome": nomes_distritos,
    "dummy": [1] * len(nomes_distritos)
})
fig = px.choropleth_mapbox(
    df,
    geojson=geojson,
    locations="ds_nome",
    featureidkey="properties.ds_nome",
    color="dummy",  # necessário, mas é um valor fixo
    color_continuous_scale=[[0, "white"], [1, "lightgrey"]],
    mapbox_style="carto-positron",
    zoom=9.8,
    center={"lat": -23.55, "lon": -46.63},
    opacity=1
)
fig.update_traces(showscale=False)
fig.update_layout(
    title="Distritos de São Paulo (apenas contorno)",
    margin={"r": 0, "t": 40, "l": 0, "b": 0}
)
fig.show()