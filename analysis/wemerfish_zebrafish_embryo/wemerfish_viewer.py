"""Interactive 3D viewer for the weMERFISH zebrafish whole-embryo data (Dash app).

Based on mytrain/scripts/deep_starmap_viewer.py (same sidebar, background-hiding,
opacity, adaptive-opacity and z-range features), adapted for the 6-embryo
weMERFISH dataset (Wan et al., Science 2026; Dryad 10.5061/dryad.j0zpc86v9):

- Adds an embryo/sample dropdown (3 stages x 2 replicates); samples are
  lazy-loaded and cached on first use, so startup is instant.
- Adds a coordinate-set dropdown (A/B stages: global_sphere is the true 3D
  spherical coordinate frame; C stage: spatial with rescaled z).
- Adds an expression-source toggle: normalized X vs raw measured obsm['X_raw'].
- The z-range slider min/max adapts to the selected sample and coordinate set.

Usage:
    conda activate ntf
    python mytrain/scripts/wemerfish_viewer.py [--port 8053]
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import anndata
import scanpy as sc
import argparse
import scipy.sparse as sp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os
import glob


# Category -> color mapping identical to eval_celltyping_deepstarmap.py
# (tab20 + tab20b + tab20c, categories sorted), so the viewer's categorical
# colors match the cell/niche-typing spatial-map figures.
CELLTYPE_COLORS = [mcolors.to_hex(c) for c in
                   (list(plt.get_cmap("tab20").colors)
                    + list(plt.get_cmap("tab20b").colors)
                    + list(plt.get_cmap("tab20c").colors))]


def celltype_color_map(categories):
    cats = sorted(str(c) for c in categories)
    return {c: CELLTYPE_COLORS[i % len(CELLTYPE_COLORS)] for i, c in enumerate(cats)}


# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Interactive 3D viewer for weMERFISH zebrafish embryos")
parser.add_argument('--data-dir', type=str,
                    default='/home/gongyuqiao/ur_annotation/NTF/mytrain/data/wemerfish_zebrafish_embryo',
                    help="Directory containing weMERFISH_measured_*.h5ad files")
parser.add_argument('--obs-cols', nargs='+',
                    default=['clusters', 'type', 'germlayer', 'tissue', 'leiden', 'stage', 'batch'],
                    help="adata.obs column names to expose as categorical options")
parser.add_argument('--port', type=int, default=8053, help="Port to serve the app on")
parser.add_argument('--base-url', type=str, default='/', help="URL path prefix, e.g. /app1/")
parser.add_argument('--title', type=str, default='weMERFISH zebrafish embryo — 3D viewer',
                    help="Title displayed in the app")
args = parser.parse_args()

# --- Sample discovery (lazy loading: h5ad files are read on first selection) ---
SAMPLE_FILES = {}
for f in sorted(glob.glob(os.path.join(args.data_dir, 'weMERFISH_measured_*.h5ad'))):
    label = os.path.basename(f).replace('weMERFISH_measured_', '').replace('.h5ad', '').replace('_rescaled_z', '')
    SAMPLE_FILES[label] = f
if not SAMPLE_FILES:
    raise SystemExit(f"No weMERFISH_measured_*.h5ad found in {args.data_dir}")
SAMPLES = list(SAMPLE_FILES.keys())
print(f"Found {len(SAMPLES)} samples in {args.data_dir}: {', '.join(SAMPLES)}")

CACHE = {}  # label -> AnnData

def load_sample(label):
    if label not in CACHE:
        print(f"Loading {SAMPLE_FILES[label]} ...")
        CACHE[label] = sc.read_h5ad(SAMPLE_FILES[label])
    return CACHE[label]

def coord_keys(adata):
    """obsm keys usable as 3D coordinates (numeric, >=3 columns; X_* matrices excluded)."""
    keys = []
    for k in adata.obsm.keys():
        if k.startswith('X_'):
            continue
        v = adata.obsm[k]
        if hasattr(v, 'shape') and len(v.shape) == 2 and v.shape[1] >= 3 and np.issubdtype(np.asarray(v).dtype, np.number):
            keys.append(k)
    return keys

def default_coord_key(adata):
    # A/B stages: global_sphere holds the true 3D spherical coordinates;
    # C stage: spatial (z already rescaled to real sample thickness).
    return 'global_sphere' if 'global_sphere' in adata.obsm else 'spatial'

def get_gene_list(adata):
    return adata.var['gene_symbol'].tolist() if 'gene_symbol' in adata.var.columns else adata.var_names.tolist()

def get_expr(adata, gene, source):
    if source == 'raw' and 'X_raw' in adata.obsm:
        j = list(adata.var_names).index(gene)
        col = np.asarray(adata.obsm['X_raw'])[:, j]
        return col.astype(np.float64).flatten()
    expr_data = adata[:, gene].X
    return expr_data.toarray().flatten() if sp.issparse(expr_data) else np.asarray(expr_data).flatten()

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

app.layout = html.Div([
    dcc.Store(id='sidebar-state', data=True),

    # --- Left control panel ---
    html.Div(id="sidebar", style=SIDEBAR_STYLE, children=[
        html.H4("Controls", className="display-4"),
        html.Hr(),
        html.Button('Hide / Show', id='toggle-sidebar', n_clicks=0, className="button-primary", style={'width': '100%', 'margin-bottom': '10px'}),

        html.Label("Embryo / sample:"),
        dcc.Dropdown(id='sample-selector',
                     options=[{'label': s, 'value': s} for s in SAMPLES],
                     value=SAMPLES[0], clearable=False),
        html.Br(),

        html.Label("Coordinate set:"),
        dcc.Dropdown(id='coord-selector', clearable=False),
        html.Br(),

        html.Label("View mode:"),
        dcc.Dropdown(
            id='view-selector',
            options=[{'label': 'Categorical', 'value': 'categorical'}, {'label': 'Gene expression (multi-select)', 'value': 'gene_expression'}],
            value='categorical', clearable=False
        ),
        html.Br(),

        html.Label("Select target (multi-select supported):"),
        dcc.Dropdown(id='item-selector', clearable=False),

        # Gene expression filter panel
        html.Div(id='gene-specific-controls', children=[
            html.Br(),
            html.Div(style={'padding': '10px', 'background-color': '#e9ecef', 'border-radius': '5px'}, children=[
                html.Strong("Expression source:"),
                dcc.RadioItems(
                    id='expr-source',
                    options=[
                        {'label': ' Normalized (X)', 'value': 'norm'},
                        {'label': ' Raw measured (obsm[X_raw])', 'value': 'raw'},
                    ],
                    value='norm', style={'margin-top': '4px', 'margin-bottom': '8px'}
                ),
                html.Hr(style={'margin-top': '4px', 'margin-bottom': '10px'}),
                html.Strong("Expression filter:"),

                html.Label("Filter reference gene (optional):", style={'margin-top': '10px', 'font-size': '13px'}),
                html.Div("If selected, the threshold below applies to this gene", style={'font-size': '11px', 'color': '#666', 'margin-bottom': '5px'}),
                dcc.Dropdown(id='filter-ref-gene', value=None,
                             placeholder="Default: filter by own expression", clearable=True),

                html.Div("Tip: when viewing multiple genes, consider hiding zero-expression cells", style={'font-size': '11px', 'color': '#666', 'margin-top': '8px'}),
                dcc.Checklist(
                    id='filter-zero',
                    options=[{'label': ' Hide cells with zero expression', 'value': 'hide_zero'}],
                    value=[],
                    style={'margin-top': '2px', 'margin-bottom': '8px'}
                ),
                html.Label("Minimum expression threshold (absolute):"),
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
        ], style={'display': 'none'}),

        html.Br(),
        html.Label("Point size:"),
        dcc.Slider(id='size-slider', min=1, max=10, step=0.5, value=3, marks={1: '1', 5: '5', 10: '10'}),
        html.Br(),

        html.Label("Global max opacity:"),
        dcc.Slider(id='opacity-slider', min=0.1, max=1, step=0.1, value=1.0, marks={0.1: '0.1', 1: '1'}),
        html.Hr(),

        html.Label("z range:"),
        dcc.RangeSlider(id='z-range', min=0, max=1, value=[0, 1],
                        tooltip={"placement": "bottom", "always_visible": True}),
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
        html.Div(id='sample-info', style={'text-align': 'center', 'color': '#666'}),
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

# Per-sample: coordinate-set options, z slider range, ref-gene options, sample info
@app.callback(
    [Output('coord-selector', 'options'),
     Output('coord-selector', 'value'),
     Output('z-range', 'min'),
     Output('z-range', 'max'),
     Output('z-range', 'value'),
     Output('z-range', 'marks'),
     Output('filter-ref-gene', 'options'),
     Output('item-selector', 'options'),
     Output('sample-info', 'children')],
    [Input('sample-selector', 'value'),
     Input('view-selector', 'value')],
    [State('coord-selector', 'value')]
)
def update_sample_controls(sample, view_mode, current_coord):
    adata = load_sample(sample)
    ck = coord_keys(adata)
    coord_opts = [{'label': k, 'value': k} for k in ck]
    coord_val = current_coord if current_coord in ck else default_coord_key(adata)

    xyz = np.asarray(adata.obsm[coord_val])[:, :3]
    z = xyz[:, 2]
    z_min, z_max = float(np.floor(z.min())), float(np.ceil(z.max()))
    if z_max <= z_min:
        z_max = z_min + 1
    step = (z_max - z_min) / 8
    marks = {round(z_min + i * step, 1): str(round(z_min + i * step, 1)) for i in range(9)}

    genes = get_gene_list(adata)
    gene_opts = [{'label': g, 'value': g} for g in genes]

    if view_mode == 'gene_expression':
        item_opts = gene_opts
    else:
        valid_obs = [c for c in args.obs_cols if c in adata.obs.columns]
        if not valid_obs:
            valid_obs = [c for c in adata.obs.columns if adata.obs[c].dtype == object or str(adata.obs[c].dtype) == 'category']
        item_opts = [{'label': c, 'value': c} for c in valid_obs]

    info = f"{adata.n_obs:,} cells × {adata.n_vars} genes"
    return coord_opts, coord_val, z_min, z_max, [z_min, z_max], marks, gene_opts, item_opts, info

@app.callback(
    [Output('item-selector', 'value'),
     Output('item-selector', 'multi'),
     Output('gene-specific-controls', 'style')],
    [Input('view-selector', 'value'),
     Input('item-selector', 'options')]
)
def update_selector_and_controls(view_mode, options):
    if not options:
        return None, False, {'display': 'none'}
    if view_mode == 'gene_expression':
        return [options[0]['value']], True, {'display': 'block'}
    return options[0]['value'], False, {'display': 'none'}

# Core plotting logic
@app.callback(
    Output('spatial-3d-plot', 'figure'),
    [Input('view-selector', 'value'),
     Input('item-selector', 'value'),
     Input('size-slider', 'value'),
     Input('opacity-slider', 'value'),
     Input('visual-options', 'value'),
     Input('z-range', 'value'),
     Input('filter-ref-gene', 'value'),
     Input('filter-zero', 'value'),
     Input('expr-threshold', 'value'),
     Input('adaptive-opacity', 'value'),
     Input('sample-selector', 'value'),
     Input('coord-selector', 'value'),
     Input('expr-source', 'value')],
    [State('spatial-3d-plot', 'relayoutData')]
)
def update_graph(view_mode, selected_item, size, opacity, visual_opts, z_range,
                 filter_ref_gene, filter_zero, expr_threshold, adaptive_opacity,
                 sample, coord_key, expr_source, relayout_data):
    if not selected_item or not sample or not coord_key:
        return go.Figure()

    adata = load_sample(sample)
    if coord_key not in adata.obsm:
        return go.Figure()

    xyz = np.asarray(adata.obsm[coord_key])[:, :3]
    df = pd.DataFrame(xyz, columns=['x', 'y', 'z'], index=adata.obs.index)
    valid_obs_cols = [c for c in args.obs_cols if c in adata.obs.columns]
    if not valid_obs_cols:
        valid_obs_cols = [c for c in adata.obs.columns if adata.obs[c].dtype == object or str(adata.obs[c].dtype) == 'category']
    if valid_obs_cols:
        df = df.join(adata.obs[valid_obs_cols].astype(str))
    gene_list = get_gene_list(adata)

    camera = relayout_data['scene.camera'] if relayout_data and 'scene.camera' in relayout_data else None
    hide_bg = 'hide_bg' in visual_opts
    axis_template = dict(
        showgrid=not hide_bg, showbackground=not hide_bg,
        showticklabels=not hide_bg, title='' if hide_bg else None, visible=not hide_bg
    )
    scene_settings = dict(aspectmode='data', xaxis=axis_template, yaxis=axis_template, zaxis=axis_template, camera=camera)
    bg_color = "rgba(0,0,0,0)" if hide_bg else "white"

    z_min_all, z_max_all = float(df['z'].min()), float(df['z'].max())
    z_mask = (df['z'].to_numpy() >= z_range[0]) & (df['z'].to_numpy() <= z_range[1])

    if view_mode == 'gene_expression':
        genes = selected_item if isinstance(selected_item, list) else [selected_item]
        genes = [g for g in genes if g in adata.var_names]
        if not genes:
            return go.Figure(layout={'title': "Selected gene(s) not found in this sample"})
        if len(genes) > 5:
            genes = genes[:5]

        color_scales = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
        traces = []

        # Pre-fetch reference gene expression for filtering
        ref_expr = None
        if filter_ref_gene and filter_ref_gene in adata.var_names:
            ref_expr = get_expr(adata, filter_ref_gene, expr_source)

        for i, gene in enumerate(genes):
            expr = get_expr(adata, gene, expr_source)
            mask_target_expr = ref_expr if ref_expr is not None else expr

            mask = z_mask.copy()
            if 'hide_zero' in filter_zero:
                mask &= (mask_target_expr > 0)
            if expr_threshold is not None and expr_threshold > 0:
                mask &= (mask_target_expr >= expr_threshold)

            if mask.sum() == 0:
                continue

            df_filtered = df[mask]
            expr_filtered = expr[mask]

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
                        color=[expr.min(), expr.max()],
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

        title_suffix = []
        if expr_source == 'raw':
            title_suffix.append("raw counts")
        if 'hide_zero' in filter_zero:
            title_suffix.append("zeros hidden")
        if expr_threshold is not None and expr_threshold > 0:
            title_suffix.append(f"threshold >= {expr_threshold}")
        if adaptive_opacity and 'adaptive' in adaptive_opacity:
            title_suffix.append("adaptive opacity")

        filter_target_name = filter_ref_gene if filter_ref_gene else "self"
        genes_str = ", ".join(genes)

        if title_suffix:
            title = f"{sample} | Gene: {genes_str} (filtered by {filter_target_name}: {', '.join(title_suffix)})"
        else:
            title = f"{sample} | Gene: {genes_str}"

        if isinstance(selected_item, list) and len(selected_item) > 5:
            title += " | Showing first 5 only"

    else:
        cat_item = selected_item[0] if isinstance(selected_item, list) else selected_item
        if cat_item not in df.columns:
            return go.Figure(layout={'title': f"Column '{cat_item}' not found in this sample"})
        plot_data = df[z_mask].copy()
        plot_data[cat_item] = plot_data[cat_item].astype(str)
        fig = px.scatter_3d(plot_data, x='x', y='y', z='z', color=cat_item,
                            color_discrete_map=celltype_color_map(df[cat_item].astype(str).unique()),
                            category_orders={cat_item: sorted(plot_data[cat_item].unique())})
        fig.update_traces(marker=dict(size=size, opacity=opacity))
        title = f"{sample} | Categorical: {cat_item}"

    if z_range[0] > z_min_all + 1e-6 or z_range[1] < z_max_all - 1e-6:
        title += f" | z {z_range[0]:.0f}–{z_range[1]:.0f}"

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
