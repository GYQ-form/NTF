"""Interactive 3D viewer for the t-CyCIF CRC1 3D atlas (Dash app).

Based on mytrain/scripts/wemerfish_viewer.py (same sidebar, background-hiding,
opacity, adaptive-opacity and z-range features), adapted for the CRC1 colorectal
cancer t-CyCIF dataset (Lin et al., Cell 2023; 25 registered slices, ~12.8M cells):

- Single-sample protein-imaging data: no sample dropdown; the
  color-by list is the 24-plex marker panel plus numeric obs (AREA, CIRC).
- Coordinate-set dropdown: 'spatial_aligned_um' (all axes in um, default;
  x/y registered across slices with MAPS sequential rigid alignment),
  'spatial_um' (all axes in um, x/y unregistered) or 'spatial'
  (unregistered pixels for x/y, um for z).
- Adds a raw / log1p intensity transform toggle (CyCIF intensities are
  highly skewed 16-bit-scale values).
- Adds a max-points downsampling control (the full dataset is ~12.8M cells,
  far beyond interactive 3D rendering; downsampling affects display only).
- Adds a z-stretch slider (display only; default 1 = true scale, since
  spatial_um puts all three axes in um; raise it to visually exaggerate the
  ~0.5 mm z span relative to the ~12 mm x/y span).
- The z-range slider filters by z in um (one value per section).

Input is the preprocessed h5ad from mytrain/scripts/preprocess_cycif_crc1.py.

Usage:
    conda activate ntf
    python mytrain/scripts/cycif_crc_viewer.py [--port 8055]
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import anndata
import argparse
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Category -> color mapping identical to eval_celltyping_cycif.py /
# eval_nichetyping_cycif.py (tab20 + tab20b + tab20c, categories sorted),
# so the viewer's categorical colors match the cell/niche-typing
# spatial-map figures.
CELLTYPE_COLORS = [mcolors.to_hex(c) for c in
                   (list(plt.get_cmap("tab20").colors)
                    + list(plt.get_cmap("tab20b").colors)
                    + list(plt.get_cmap("tab20c").colors))]


def celltype_color_map(categories):
    cats = sorted(str(c) for c in categories)
    return {c: CELLTYPE_COLORS[i % len(CELLTYPE_COLORS)] for i, c in enumerate(cats)}

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Interactive 3D viewer for the t-CyCIF CRC1 3D atlas")
parser.add_argument('-i', '--input_path', type=str,
                    default='/home/gongyuqiao/ur_annotation/NTF/mytrain/data/cycif_crc1_3d/processed/cycif_crc1.h5ad',
                    help="Path to the preprocessed cycif_crc1 h5ad")
parser.add_argument('--obs-cols', nargs='+', default=['slice', 'cell_type', 'cell_type_main'],
                    help="adata.obs column names to expose as categorical options")
parser.add_argument('--port', type=int, default=8055, help="Port to serve the app on")
parser.add_argument('--base-url', type=str, default='/', help="URL path prefix, e.g. /app1/")
parser.add_argument('--title', type=str, default='t-CyCIF CRC1 3D atlas — 3D viewer',
                    help="Title displayed in the app")
args = parser.parse_args()

# --- Data loading (single h5ad, loaded once at startup) ---
print(f"Loading {args.input_path} ...")
adata = anndata.read_h5ad(args.input_path)
print(f"Loaded: {adata.n_obs:,} cells x {adata.n_vars} markers")

# Coordinate sets: numeric obsm matrices with >= 3 columns
COORD_KEYS = [k for k in adata.obsm.keys()
              if not k.startswith('X_') and hasattr(adata.obsm[k], 'shape')
              and len(adata.obsm[k].shape) == 2 and adata.obsm[k].shape[1] >= 3
              and np.issubdtype(np.asarray(adata.obsm[k]).dtype, np.number)]
DEFAULT_COORD = ('spatial_aligned_um' if 'spatial_aligned_um' in COORD_KEYS
                 else 'spatial_um' if 'spatial_um' in COORD_KEYS else COORD_KEYS[0])
COORD_OPTS = [{'label': k, 'value': k} for k in COORD_KEYS]

def get_xyz(coord_key):
    return np.asarray(adata.obsm[coord_key])[:, :3].astype(np.float64)

# z is in um in every coordinate set, so the z slider range is fixed
Z_MIN, Z_MAX = float(get_xyz(DEFAULT_COORD)[:, 2].min()), float(get_xyz(DEFAULT_COORD)[:, 2].max())

# Order slice categories numerically (2, 7, 14, ...) instead of lexicographically
if 'slice_id' in adata.obs.columns:
    adata.obs['slice'] = pd.Categorical(adata.obs['slice_id'].astype(str),
                                        categories=[str(s) for s in sorted(adata.obs['slice_id'].unique())])

MARKER_LIST = adata.var_names.tolist()
NUMERIC_OBS = [c for c in adata.obs.columns if pd.api.types.is_numeric_dtype(adata.obs[c])]
FEATURE_LIST = MARKER_LIST + NUMERIC_OBS
FEATURE_OPTS = [{'label': f, 'value': f} for f in FEATURE_LIST]

valid_obs_cols = [c for c in args.obs_cols if c in adata.obs.columns]
if not valid_obs_cols:
    valid_obs_cols = [c for c in adata.obs.columns
                      if adata.obs[c].dtype == object or isinstance(adata.obs[c].dtype, pd.CategoricalDtype)]
CAT_OPTS = [{'label': c, 'value': c} for c in valid_obs_cols]

RNG = np.random.default_rng(0)


def get_expr(name):
    """Raw values for a marker (from X) or a numeric obs column."""
    if name in adata.var_names:
        expr_data = adata[:, name].X
        col = expr_data.toarray().flatten() if sp.issparse(expr_data) else np.asarray(expr_data).flatten()
        return col.astype(np.float64)
    return adata.obs[name].to_numpy(np.float64)


def subsample_idx(n, max_points):
    if n <= max_points:
        return np.arange(n)
    return RNG.choice(n, size=max_points, replace=False)

# --- Dash App ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
                url_base_pathname=args.base_url)

SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0, "width": "20rem",
    "padding": "2rem 1rem", "background-color": "#f8f9fa", "overflow-y": "auto",
    "transition": "margin-left .5s", "z-index": 1000, "box-shadow": "2px 0 5px rgba(0,0,0,0.1)"
}
CONTENT_STYLE = {
    "margin-left": "22rem", "margin-right": "2rem", "padding": "2rem 1rem", "transition": "margin-left .5s"
}

z_step = (Z_MAX - Z_MIN) / 8
z_marks = {round(Z_MIN + i * z_step): str(round(Z_MIN + i * z_step)) for i in range(9)}

app.layout = html.Div([
    dcc.Store(id='sidebar-state', data=True),

    # --- Left control panel ---
    html.Div(id="sidebar", style=SIDEBAR_STYLE, children=[
        html.H4("Controls", className="display-4"),
        html.Hr(),
        html.Button('Hide / Show', id='toggle-sidebar', n_clicks=0, className="button-primary", style={'width': '100%', 'margin-bottom': '10px'}),

        html.Label("View mode:"),
        dcc.Dropdown(
            id='view-selector',
            options=[{'label': 'Categorical', 'value': 'categorical'}, {'label': 'Marker expression (multi-select)', 'value': 'gene_expression'}],
            value='gene_expression', clearable=False
        ),
        html.Br(),

        html.Label("Coordinate set:"),
        dcc.Dropdown(id='coord-selector', options=COORD_OPTS, value=DEFAULT_COORD, clearable=False),
        html.Br(),

        html.Label("Select target (multi-select supported):"),
        dcc.Dropdown(id='item-selector', clearable=False),

        # Marker expression filter panel
        html.Div(id='gene-specific-controls', children=[
            html.Br(),
            html.Div(style={'padding': '10px', 'background-color': '#e9ecef', 'border-radius': '5px'}, children=[
                html.Strong("Intensity transform:"),
                dcc.RadioItems(
                    id='expr-transform',
                    options=[
                        {'label': ' log1p', 'value': 'log1p'},
                        {'label': ' Raw', 'value': 'raw'},
                    ],
                    value='log1p', style={'margin-top': '4px', 'margin-bottom': '8px'}
                ),
                html.Hr(style={'margin-top': '4px', 'margin-bottom': '10px'}),
                html.Strong("Expression filter:"),

                html.Label("Filter reference marker (optional):", style={'margin-top': '10px', 'font-size': '13px'}),
                html.Div("If selected, the threshold below applies to this marker", style={'font-size': '11px', 'color': '#666', 'margin-bottom': '5px'}),
                dcc.Dropdown(id='filter-ref-gene', options=FEATURE_OPTS, value=None,
                             placeholder="Default: filter by own expression", clearable=True),

                html.Div("Tip: when viewing multiple markers, consider hiding zero-expression cells", style={'font-size': '11px', 'color': '#666', 'margin-top': '8px'}),
                dcc.Checklist(
                    id='filter-zero',
                    options=[{'label': ' Hide cells with zero expression', 'value': 'hide_zero'}],
                    value=[],
                    style={'margin-top': '2px', 'margin-bottom': '8px'}
                ),
                html.Label("Minimum expression threshold (absolute, on transformed scale):"),
                dcc.Input(id='expr-threshold', type='number', value=0.0, step=0.1, style={'width': '100%'}),

                # Adaptive opacity
                html.Hr(style={'margin-top': '10px', 'margin-bottom': '10px'}),
                html.Strong("Advanced rendering:"),
                dcc.Checklist(
                    id='adaptive-opacity',
                    options=[{'label': ' Adaptive opacity (lower expression = more transparent)', 'value': 'adaptive'}],
                    value=[],
                    style={'margin-top': '5px', 'margin-bottom': '5px'}
                )
            ])
        ], style={'display': 'block'}),

        html.Br(),
        html.Label("Max points displayed (random downsampling):"),
        dcc.Dropdown(
            id='max-points',
            options=[{'label': f'{v // 1000}k', 'value': v} for v in [50000, 100000, 200000, 500000]],
            value=200000, clearable=False
        ),
        html.Br(),

        html.Label("Point size:"),
        dcc.Slider(id='size-slider', min=1, max=10, step=0.5, value=2, marks={1: '1', 5: '5', 10: '10'}),
        html.Br(),

        html.Label("Global max opacity:"),
        dcc.Slider(id='opacity-slider', min=0.1, max=1, step=0.1, value=1.0, marks={0.1: '0.1', 1: '1'}),
        html.Hr(),

        html.Label("z range (µm):"),
        dcc.RangeSlider(id='z-range', min=Z_MIN, max=Z_MAX, value=[Z_MIN, Z_MAX], marks=z_marks,
                        tooltip={"placement": "bottom", "always_visible": True}),
        html.Br(),

        html.Label("z stretch (display only; 1 = true scale):"),
        dcc.Slider(id='z-stretch', min=1, max=50, step=1, value=1,
                   marks={1: '1', 10: '10', 25: '25', 50: '50'}),
        html.Hr(),

        html.Label("Visual options:"),
        dcc.Checklist(
            id='visual-options',
            options=[
                {'label': ' Hide axis background (immersive mode)', 'value': 'hide_bg'},
            ],
            value=[]
        )
    ]),

    # --- Right content area ---
    html.Div(id="page-content", style=CONTENT_STYLE, children=[
        html.H2(args.title, style={'text-align': 'center'}),
        html.Div(f"{adata.n_obs:,} cells × {adata.n_vars} markers, 25 sections, z {Z_MIN:.0f}–{Z_MAX:.0f} µm",
                 style={'text-align': 'center', 'color': '#666'}),
        dcc.Loading(
            type="circle",
            children=dcc.Graph(id='spatial-3d-plot', style={'height': '82vh'})
        )
    ])
])

# --- Callbacks ---

@app.callback(
    [Output("sidebar", "style"), Output("page-content", "style")],
    [Input("toggle-sidebar", "n_clicks")],
    [State("sidebar", "style"), State("page-content", "style")]
)
def toggle_sidebar(n_clicks, sidebar_style, content_style):
    if n_clicks % 2 == 1:
        sidebar_style['margin-left'] = "-20rem"
        content_style['margin-left'] = "2rem"
    else:
        sidebar_style['margin-left'] = "0"
        content_style['margin-left'] = "22rem"
    return sidebar_style, content_style


@app.callback(
    [Output('item-selector', 'options'),
     Output('item-selector', 'value'),
     Output('item-selector', 'multi'),
     Output('gene-specific-controls', 'style')],
    [Input('view-selector', 'value')]
)
def update_selector_and_controls(view_mode):
    if view_mode == 'gene_expression':
        return FEATURE_OPTS, [FEATURE_LIST[0]], True, {'display': 'block'}
    return CAT_OPTS, (CAT_OPTS[0]['value'] if CAT_OPTS else None), False, {'display': 'none'}


# Core plotting logic
@app.callback(
    Output('spatial-3d-plot', 'figure'),
    [Input('view-selector', 'value'),
     Input('item-selector', 'value'),
     Input('size-slider', 'value'),
     Input('opacity-slider', 'value'),
     Input('visual-options', 'value'),
     Input('z-range', 'value'),
     Input('z-stretch', 'value'),
     Input('max-points', 'value'),
     Input('filter-ref-gene', 'value'),
     Input('filter-zero', 'value'),
     Input('expr-threshold', 'value'),
     Input('adaptive-opacity', 'value'),
     Input('expr-transform', 'value'),
     Input('coord-selector', 'value')],
    [State('spatial-3d-plot', 'relayoutData')]
)
def update_graph(view_mode, selected_item, size, opacity, visual_opts, z_range, z_stretch,
                 max_points, filter_ref_gene, filter_zero, expr_threshold, adaptive_opacity,
                 expr_transform, coord_key, relayout_data):
    if not selected_item or not coord_key or coord_key not in adata.obsm:
        return go.Figure()

    xyz = get_xyz(coord_key)
    z_stretch = z_stretch or 1
    df = pd.DataFrame({'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2] * z_stretch})

    camera = relayout_data['scene.camera'] if relayout_data and 'scene.camera' in relayout_data else None
    hide_bg = 'hide_bg' in visual_opts
    axis_template = dict(
        showgrid=not hide_bg, showbackground=not hide_bg,
        showticklabels=not hide_bg, title='' if hide_bg else None, visible=not hide_bg
    )
    scene_settings = dict(aspectmode='data', xaxis=axis_template, yaxis=axis_template, zaxis=axis_template, camera=camera)
    bg_color = "rgba(0,0,0,0)" if hide_bg else "white"

    z_mask = (xyz[:, 2] >= z_range[0]) & (xyz[:, 2] <= z_range[1])

    def transform(v):
        return np.log1p(v) if expr_transform == 'log1p' else v

    if view_mode == 'gene_expression':
        genes = selected_item if isinstance(selected_item, list) else [selected_item]
        genes = [g for g in genes if g in FEATURE_LIST]
        if not genes:
            return go.Figure(layout={'title': "Selected marker(s) not found"})
        if len(genes) > 5:
            genes = genes[:5]

        color_scales = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
        traces = []

        # Pre-fetch reference marker expression for filtering
        ref_expr = None
        if filter_ref_gene and filter_ref_gene in FEATURE_LIST:
            ref_expr = transform(get_expr(filter_ref_gene))

        for i, gene in enumerate(genes):
            expr = transform(get_expr(gene))
            mask_target_expr = ref_expr if ref_expr is not None else expr

            mask = z_mask.copy()
            if 'hide_zero' in filter_zero:
                mask &= (mask_target_expr > 0)
            if expr_threshold is not None and expr_threshold > 0:
                mask &= (mask_target_expr >= expr_threshold)

            if mask.sum() == 0:
                continue

            keep = subsample_idx(int(mask.sum()), max_points)
            df_filtered = df[mask].iloc[keep]
            expr_filtered = expr[mask][keep]

            cscale = 'Viridis' if len(genes) == 1 else color_scales[i % len(color_scales)]
            cbar_x = 1.0 if len(genes) == 1 else (1.02 + (i * 0.08))

            # Adaptive opacity rendering
            if adaptive_opacity and 'adaptive' in adaptive_opacity:
                e_min = expr_filtered.min()
                e_max = expr_filtered.max()

                if e_max > e_min:
                    norm_expr = (expr_filtered - e_min) / (e_max - e_min)
                else:
                    norm_expr = np.ones_like(expr_filtered)

                dynamic_alpha = norm_expr * opacity

                cmap_name = cscale.lower() if cscale == 'Viridis' else cscale
                cmap = plt.get_cmap(cmap_name)
                rgba_matrix = cmap(norm_expr)
                rgba_matrix[:, 3] = dynamic_alpha

                color_array = [f'rgba({int(r*255)},{int(g*255)},{int(b*255)},{a:.3f})' for r, g, b, a in rgba_matrix]

                # Real 3D scatter trace (colour and opacity encoded in rgba strings)
                traces.append(go.Scatter3d(
                    x=df_filtered['x'], y=df_filtered['y'], z=df_filtered['z'],
                    mode='markers', name=gene,
                    marker=dict(size=size, color=color_array),
                    hovertext=[f'{gene}: {v:.3f}' for v in expr_filtered], hoverinfo='text'
                ))

                # Hidden dummy trace solely for attaching the colorbar
                traces.append(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers', showlegend=False, hoverinfo='none',
                    marker=dict(
                        size=0,
                        color=[expr_filtered.min(), expr_filtered.max()],
                        colorscale=cscale,
                        colorbar=dict(title=gene, x=cbar_x, thickness=12, len=0.7),
                        showscale=True
                    )
                ))

            else:
                # Standard uniform opacity rendering
                traces.append(go.Scatter3d(
                    x=df_filtered['x'], y=df_filtered['y'], z=df_filtered['z'],
                    mode='markers',
                    name=gene,
                    marker=dict(
                        size=size, opacity=opacity,
                        color=expr_filtered, colorscale=cscale,
                        colorbar=dict(title=gene, x=cbar_x, thickness=12, len=0.7),
                        showscale=True
                    ),
                    hovertext=[f'{gene}: {v:.3f}' for v in expr_filtered], hoverinfo='text'
                ))

        if not traces:
            return go.Figure(layout={'title': "No data points match the current filter settings"})

        fig = go.Figure(data=traces)

        title_suffix = [expr_transform]
        if 'hide_zero' in filter_zero:
            title_suffix.append("zeros hidden")
        if expr_threshold is not None and expr_threshold > 0:
            title_suffix.append(f"threshold >= {expr_threshold}")
        if adaptive_opacity and 'adaptive' in adaptive_opacity:
            title_suffix.append("adaptive opacity")

        filter_target_name = filter_ref_gene if filter_ref_gene else "self"
        genes_str = ", ".join(genes)
        title = f"Marker: {genes_str} ({', '.join(title_suffix)}; filtered by {filter_target_name})"

        if isinstance(selected_item, list) and len(selected_item) > 5:
            title += " | Showing first 5 only"

    else:
        cat_item = selected_item[0] if isinstance(selected_item, list) else selected_item
        if cat_item not in adata.obs.columns:
            return go.Figure(layout={'title': f"Column '{cat_item}' not found"})
        df['cat'] = adata.obs[cat_item].astype(str).to_numpy()
        mask_idx = np.where(z_mask)[0]
        keep = mask_idx[subsample_idx(len(mask_idx), max_points)]
        plot_data = df.iloc[keep]
        fig = px.scatter_3d(plot_data, x='x', y='y', z='z', color='cat',
                            color_discrete_map=celltype_color_map(df['cat'].unique()),
                            category_orders={'cat': sorted(plot_data['cat'].unique(),
                                                           key=lambda s: (0, float(s)) if s.replace('.', '', 1).isdigit() else (1, s))})
        fig.update_traces(marker=dict(size=size, opacity=opacity))
        title = f"Categorical: {cat_item}"

    if z_range[0] > Z_MIN + 1e-6 or z_range[1] < Z_MAX - 1e-6:
        title += f" | z {z_range[0]:.0f}–{z_range[1]:.0f} µm"

    fig.update_layout(
        title=title, scene=scene_settings,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor=bg_color,
        showlegend=True,
        legend=dict(font=dict(size=14), itemsizing='constant', yanchor="top", y=0.9, xanchor="left", x=0.05)
    )
    return fig


if __name__ == '__main__':
    app.run(debug=False, port=args.port, host='0.0.0.0')
