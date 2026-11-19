"""Interactive 3D viewer for the Deep-STARmap mouse brain data (Dash app).

Based on scripts/3D_visulization.py (same sidebar, background-hiding, opacity
and adaptive-opacity features), with a z-layer range slider added for the
208 optical layers of this dataset.

Usage:
    conda activate ntf
    python mytrain/scripts/deep_starmap_viewer.py [--port 8051]
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

# Category -> color mapping identical to eval_celltyping_deepstarmap.py
# (tab20 + tab20b + tab20c, categories sorted), so the viewer's categorical
# colors match the cell-typing spatial-map figures.
CELLTYPE_COLORS = [mcolors.to_hex(c) for c in
                   (list(plt.get_cmap("tab20").colors)
                    + list(plt.get_cmap("tab20b").colors)
                    + list(plt.get_cmap("tab20c").colors))]


def celltype_color_map(categories):
    cats = sorted(str(c) for c in categories)
    return {c: CELLTYPE_COLORS[i % len(CELLTYPE_COLORS)] for i, c in enumerate(cats)}

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Interactive 3D spatial transcriptomics visualisation")
parser.add_argument('--obs-cols', nargs='+',
                    default=['Harmony_labels', 'FUSEmap_main_level', 'FUSEmap_sub_level', 'slice_id'],
                    help="adata.obs column names to expose as categorical options")
parser.add_argument('--port', type=int, default=8051, help="Port to serve the app on")
parser.add_argument('--base-url', type=str, default='/', help="URL path prefix, e.g. /app1/")
parser.add_argument('--title', type=str, default='Deep-STARmap mouse brain — 3D viewer',
                    help="Title displayed in the app")
parser.add_argument('-i', "--input_path", type=str,
                    default='/home/gongyuqiao/ur_annotation/NTF/mytrain/data/deep_starmap_brain/processed/deep_starmap_brain.h5ad',
                    help="Path to the .h5ad input file")
args = parser.parse_args()

# --- Data loading ---
print(f"Loading data: {args.input_path}")
adata_orig = sc.read_h5ad(args.input_path)

# Down-sample large datasets for interactive performance
SAMPLING_THRESHOLD = 100000
if adata_orig.n_obs > SAMPLING_THRESHOLD:
    adata = sc.pp.subsample(adata_orig, n_obs=SAMPLING_THRESHOLD, copy=True)
    print(f"Dataset exceeds {SAMPLING_THRESHOLD} cells; automatically down-sampled.")
else:
    adata = adata_orig

df = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y', 'z'], index=adata.obs.index)
valid_obs_cols = [col for col in args.obs_cols if col in adata.obs.columns]
if not valid_obs_cols and 'cell_type' in adata.obs: valid_obs_cols = ['cell_type']
if valid_obs_cols: df = df.join(adata.obs[valid_obs_cols].astype(str))

adata.var_names = adata.var.gene_symbol if 'gene_symbol' in adata.var.columns else adata.var_names
gene_list = adata.var_names.tolist()

Z_MIN, Z_MAX = int(df['z'].min()), int(df['z'].max())

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
                html.Strong("Expression filter:"),

                html.Label("Filter reference gene (optional):", style={'margin-top': '10px', 'font-size': '13px'}),
                html.Div("If selected, the threshold below applies to this gene", style={'font-size': '11px', 'color': '#666', 'margin-bottom': '5px'}),
                dcc.Dropdown(
                    id='filter-ref-gene',
                    options=[{'label': g, 'value': g} for g in gene_list],
                    value=None,
                    placeholder="Default: filter by own expression",
                    clearable=True
                ),

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

        html.Label(f"z-layer range ({Z_MIN}-{Z_MAX}):"),
        dcc.RangeSlider(id='z-range', min=Z_MIN, max=Z_MAX, value=[Z_MIN, Z_MAX],
                        marks={z: str(z) for z in range(Z_MIN, Z_MAX + 1, 26)},
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
        dcc.Loading(
            type="circle",
            children=dcc.Graph(id='spatial-3d-plot', style={'height': '85vh'})
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
        opts = [{'label': g, 'value': g} for g in gene_list]
        val = [gene_list[0]] if gene_list else []
        return opts, val, True, {'display': 'block'}
    else:
        opts = [{'label': c, 'value': c} for c in valid_obs_cols]
        val = valid_obs_cols[0] if valid_obs_cols else None
        return opts, val, False, {'display': 'none'}

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
     Input('adaptive-opacity', 'value')],
    [State('spatial-3d-plot', 'relayoutData')]
)
def update_graph(view_mode, selected_item, size, opacity, visual_opts, z_range,
                 filter_ref_gene, filter_zero, expr_threshold, adaptive_opacity, relayout_data):
    if not selected_item: return go.Figure()

    camera = relayout_data['scene.camera'] if relayout_data and 'scene.camera' in relayout_data else None
    hide_bg = 'hide_bg' in visual_opts
    axis_template = dict(
        showgrid=not hide_bg, showbackground=not hide_bg,
        showticklabels=not hide_bg, title='' if hide_bg else None, visible=not hide_bg
    )
    scene_settings = dict(aspectmode='data', xaxis=axis_template, yaxis=axis_template, zaxis=axis_template, camera=camera)
    bg_color = "rgba(0,0,0,0)" if hide_bg else "white"

    z_mask = (df['z'].to_numpy() >= z_range[0]) & (df['z'].to_numpy() <= z_range[1])

    if view_mode == 'gene_expression':
        genes = selected_item if isinstance(selected_item, list) else [selected_item]
        if len(genes) > 5: genes = genes[:5]

        color_scales = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']
        traces = []

        # Pre-fetch reference gene expression for filtering
        ref_expr = None
        if filter_ref_gene and filter_ref_gene in adata.var_names:
            ref_data = adata[:, filter_ref_gene].X
            ref_expr = ref_data.toarray().flatten() if sp.issparse(ref_data) else np.asarray(ref_data).flatten()

        for i, gene in enumerate(genes):
            # Target gene expression
            expr_data = adata[:, gene].X
            expr = expr_data.toarray().flatten() if sp.issparse(expr_data) else np.asarray(expr_data).flatten()

            mask_target_expr = ref_expr if ref_expr is not None else expr

            mask = z_mask.copy()
            if 'hide_zero' in filter_zero: mask &= (mask_target_expr > 0)
            if expr_threshold is not None and expr_threshold > 0: mask &= (mask_target_expr >= expr_threshold)

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
        if 'hide_zero' in filter_zero: title_suffix.append("zeros hidden")
        if expr_threshold is not None and expr_threshold > 0: title_suffix.append(f"threshold >= {expr_threshold}")
        if adaptive_opacity and 'adaptive' in adaptive_opacity: title_suffix.append("adaptive opacity")

        filter_target_name = filter_ref_gene if filter_ref_gene else "self"
        genes_str = ", ".join(genes)

        if title_suffix:
            title = f"Gene: {genes_str} (filtered by {filter_target_name}: {', '.join(title_suffix)})"
        else:
            title = f"Gene: {genes_str}"

        if len(selected_item) > 5: title += " | Showing first 5 only"

    else:
        cat_item = selected_item[0] if isinstance(selected_item, list) else selected_item
        plot_data = df[z_mask].copy()
        plot_data[cat_item] = plot_data[cat_item].astype(str)

        fig = px.scatter_3d(plot_data, x='x', y='y', z='z', color=cat_item,
                            color_discrete_map=celltype_color_map(plot_data[cat_item].unique()),
                            category_orders={cat_item: sorted(plot_data[cat_item].unique())})
        fig.update_traces(marker=dict(size=size, opacity=opacity))
        title = f"Categorical: {cat_item}"

    if z_range[0] > Z_MIN or z_range[1] < Z_MAX:
        title += f" | z {z_range[0]}-{z_range[1]}"

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
