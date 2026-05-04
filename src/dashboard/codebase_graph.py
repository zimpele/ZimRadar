"""Parse ZimRadar source tree into an interactive architecture graph.

Extracts modules, functions, and their descriptions (from docstrings or
Gemma2-generated), then builds a Plotly network figure with a layered
left-to-right layout reflecting the data-flow architecture.
"""

import ast
import json
import pathlib
import re
import textwrap

import httpx
import networkx as nx
import plotly.graph_objects as go

SRC_ROOT = pathlib.Path(__file__).parent.parent  # ZimRadar/src/
CACHE_FILE = pathlib.Path(__file__).parent.parent.parent / ".omc" / "function_descriptions.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma2:9b"

# Left-to-right data flow: column x-index per layer
LAYER_CFG = {
    "root": {"color": "#607D8B", "x": 0, "label": "Config"},
    "ingestion": {"color": "#2196F3", "x": 1, "label": "Ingestion"},
    "storage": {"color": "#9E9E9E", "x": 2, "label": "Storage"},
    "pipeline": {"color": "#4CAF50", "x": 3, "label": "Pipeline"},
    "rag": {"color": "#F44336", "x": 3, "label": "RAG"},
    "agents": {"color": "#FF9800", "x": 4, "label": "Agents"},
    "api": {"color": "#00BCD4", "x": 5, "label": "API"},
    "dashboard": {"color": "#9C27B0", "x": 5, "label": "Dashboard"},
}


# ── AST parsing ───────────────────────────────────────────────────────────────


def _get_layer(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC_ROOT)
    return rel.parts[0] if len(rel.parts) > 1 else "root"


def _extract_functions(path: pathlib.Path) -> list[dict]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    results = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        args: list[str] = []
        if kind == "function":
            args = [a.arg for a in node.args.args if a.arg != "self"]
        results.append(
            {
                "name": node.name,
                "kind": kind,
                "args": args,
                "docstring": ast.get_docstring(node) or "",
                "lineno": node.lineno,
            }
        )
    return results


def _extract_src_imports(path: pathlib.Path) -> list[str]:
    """Return module ids (e.g. 'ingestion/fema') that this file imports from src.*"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    ids = set()
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    mod = alias.name
        if mod and mod.startswith("src."):
            parts = mod.split(".")[1:]  # drop "src"
            if parts:
                ids.add("/".join(parts))
    return list(ids)


# ── Description generation ────────────────────────────────────────────────────


def _clean_docstring(doc: str) -> str:
    first_para = doc.split("\n\n")[0].strip()
    return textwrap.shorten(first_para.replace("\n", " "), width=220, placeholder="…")


def _name_to_hint(name: str) -> str:
    words = re.sub(r"_+", " ", name.lstrip("_")).strip()
    return words.capitalize()


def generate_description_via_llm(name: str, args: list[str], module: str) -> str:
    sig = f"{name}({', '.join(args[:4])}{'...' if len(args) > 4 else ''})"
    prompt = (
        f"You are documenting a Python project. In exactly one short sentence "
        f"(≤20 words), describe what this function does. Start with a verb.\n"
        f"Function signature: {sig}\n"
        f"Module: {module}\n"
        f"One-sentence description:"
    )
    try:
        resp = httpx.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=45.0,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return textwrap.shorten(text, width=220, placeholder="…")
    except Exception:
        return _name_to_hint(name)


# ── Graph construction ────────────────────────────────────────────────────────


def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    except Exception:
        pass


def build_graph_data(use_llm: bool = False) -> dict:
    """
    Scan src/, extract modules + functions, build node/edge data.

    Returns {"nodes": [...], "edges": [...], "cache_hits": int, "llm_calls": int}
    """
    cache = _load_cache()
    cache_hits = 0
    llm_calls = 0

    py_files = sorted(
        f for f in SRC_ROOT.rglob("*.py") if f.name != "__init__.py" and ".pyc" not in str(f)
    )

    # Group by layer, compute positions
    by_layer: dict[str, list[pathlib.Path]] = {}
    for f in py_files:
        by_layer.setdefault(_get_layer(f), []).append(f)

    G = nx.DiGraph()
    node_meta: dict[str, dict] = {}

    for layer, files in sorted(by_layer.items()):
        cfg = LAYER_CFG.get(layer, {"color": "#888", "x": 6, "label": layer.title()})
        x_pos = cfg["x"] * 3.5

        for i, filepath in enumerate(files):
            mod_id = str(filepath.relative_to(SRC_ROOT).with_suffix(""))
            mod_label = filepath.stem

            funcs = _extract_functions(filepath)
            func_entries = []
            for fn in funcs:
                cache_key = f"{mod_id}::{fn['name']}"
                if cache_key in cache:
                    desc = cache[cache_key]
                    cache_hits += 1
                elif fn["docstring"]:
                    desc = _clean_docstring(fn["docstring"])
                    cache[cache_key] = desc
                elif use_llm:
                    desc = generate_description_via_llm(fn["name"], fn["args"], mod_label)
                    cache[cache_key] = desc
                    llm_calls += 1
                else:
                    desc = _name_to_hint(fn["name"])
                func_entries.append(
                    {
                        "name": fn["name"],
                        "kind": fn["kind"],
                        "args": fn["args"],
                        "description": desc,
                    }
                )

            y_pos = (i - len(files) / 2) * 2.2
            G.add_node(mod_id)
            node_meta[mod_id] = {
                "id": mod_id,
                "label": mod_label,
                "layer": layer,
                "layer_label": cfg["label"],
                "color": cfg["color"],
                "x": x_pos,
                "y": y_pos,
                "functions": func_entries,
            }

            for imp in _extract_src_imports(filepath):
                if imp in node_meta or True:  # add edge even if target not yet seen
                    G.add_edge(mod_id, imp)

    _save_cache(cache)

    nodes = list(node_meta.values())
    edges = [{"source": u, "target": v} for u, v in G.edges() if u in node_meta and v in node_meta]

    return {
        "nodes": nodes,
        "edges": edges,
        "cache_hits": cache_hits,
        "llm_calls": llm_calls,
    }


# ── Plotly figure ─────────────────────────────────────────────────────────────


def _hover_text(node: dict) -> str:
    funcs = node["functions"]
    lines = [f"<b>{node['label']}</b>  <i>({node['layer_label']} layer)</i>", ""]
    if funcs:
        for fn in funcs[:12]:
            icon = "🔷" if fn["kind"] == "class" else "🔹"
            args_str = f"({', '.join(fn['args'][:3])}{'…' if len(fn['args']) > 3 else ''})"
            lines.append(f"{icon} <b>{fn['name']}</b>{args_str}")
            lines.append(f"   <i>{fn['description']}</i>")
            lines.append("")
        if len(funcs) > 12:
            lines.append(f"  … and {len(funcs) - 12} more")
    else:
        lines.append("<i>No public functions</i>")
    return "<br>".join(lines)


def build_plotly_figure(graph_data: dict) -> go.Figure:
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    pos = {n["id"]: (n["x"], n["y"]) for n in nodes}

    # Edge traces (one per edge for clean rendering)
    edge_x, edge_y = [], []
    for e in edges:
        x0, y0 = pos.get(e["source"], (0, 0))
        x1, y1 = pos.get(e["target"], (0, 0))
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="#cccccc"),
        hoverinfo="none",
        showlegend=False,
    )

    # Group nodes by layer for legend
    layer_traces = []
    by_layer: dict[str, list] = {}
    for n in nodes:
        by_layer.setdefault(n["layer"], []).append(n)

    for layer, layer_nodes in sorted(by_layer.items()):
        cfg = LAYER_CFG.get(layer, {"color": "#888", "label": layer.title()})
        xs = [n["x"] for n in layer_nodes]
        ys = [n["y"] for n in layer_nodes]
        texts = [n["label"] for n in layer_nodes]
        hovers = [_hover_text(n) for n in layer_nodes]
        sizes = [20 + min(len(n["functions"]) * 2, 24) for n in layer_nodes]

        layer_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                name=cfg["label"],
                text=texts,
                textposition="bottom center",
                textfont=dict(size=11),
                marker=dict(
                    size=sizes,
                    color=cfg["color"],
                    line=dict(width=1.5, color="white"),
                    opacity=0.9,
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hovers,
            )
        )

    # Layer label annotations
    layer_labels = {}
    for n in nodes:
        layer_labels.setdefault(n["layer"], (n["x"], n["layer_label"]))

    annotations = [
        dict(
            x=x,
            y=max(n["y"] for n in nodes if n["layer"] == layer) + 1.5,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=13, color=LAYER_CFG.get(layer, {}).get("color", "#888")),
            xanchor="center",
        )
        for layer, (x, label) in layer_labels.items()
    ]

    fig = go.Figure(
        data=[edge_trace] + layer_traces,
        layout=go.Layout(
            title=dict(
                text="ZimRadar — Architecture Overview",
                font=dict(size=18),
                x=0.5,
            ),
            showlegend=True,
            hovermode="closest",
            annotations=annotations,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=680,
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.08,
                xanchor="center",
                x=0.5,
            ),
        ),
    )
    return fig
