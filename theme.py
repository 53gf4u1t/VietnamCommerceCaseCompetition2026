"""Central Catppuccin theme manager (`Mocha` dark mode & `Latte` light mode) for all 5 CaseComp modules."""

from __future__ import annotations

import os
import matplotlib.colors as mcolors
import matplotlib.figure as mfigure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.collections import Collection
import numpy as np

# Default theme if not overridden via CLI (`--theme Mocha` or `--theme Latte`) or env var `PICK_THEME`
_ACTIVE_THEME: str = os.environ.get("PICK_THEME", "Mocha").strip().capitalize()
if _ACTIVE_THEME not in ("Mocha", "Latte"):
    _ACTIVE_THEME = "Mocha"

_ORIG_SAVEFIG = mfigure.Figure.savefig
_HOOK_INSTALLED = False

# ==============================================================================
# CATPPUCCIN MOCHA (DARK) -> CATPPUCCIN LATTE (LIGHT) EXACT SEMANTIC MAPPINGS
# ==============================================================================

# 1. Surface / Background / Fill colors (Patch facecolors, Axes/Figure backgrounds, Table cells)
FACE_MOCHA_TO_LATTE: dict[str, str] = {
    # Core Catppuccin surfaces
    "#1e1e2e": "#eff1f5",  # Base (Figure bg, even table row, callout box bg)
    "#181825": "#e6e9ef",  # Mantle (Axes bg, odd table row)
    "#11111b": "#dce0e8",  # Crust (Table/Gantt top banner fill)
    "#313244": "#ccd0da",  # Surface0 (Subheader bg, weekend shading, KPI box bg)
    "#45475a": "#bcc0cc",  # Surface1 (Borders, dividers)
    "#585b70": "#acb0be",  # Surface2 (Dimmed baseline bars)
    "#6c7086": "#9ca0b0",  # Overlay0
    "#7f849c": "#8c8fa1",  # Overlay1 (Baseline bars/series)
    "#9399b2": "#7c7f93",  # Overlay2
    "#a6adc8": "#6c6f85",  # Subtext0
    "#bac2de": "#5c5f77",  # Subtext1
    "#cdd6f4": "#4c4f69",  # Text
    # Tinted table & Gantt row/column backgrounds
    "#162038": "#1e3a8a",  # Major Statement Banner bg (Deep Executive Navy in Latte)
    "#2b3048": "#334155",  # Top Table Column Header bg (Dark Slate in Latte)
    "#262b40": "#d6deeb",  # Subsection Group Header bg (Cool Slate-Blue tint in Latte)
    "#252638": "#dce2ee",  # Subtotal row bg
    "#232436": "#e2e5ee",  # Memo row bg
    "#1f332c": "#d6eedb",  # Total highlight row bg (soft Latte Green)
    "#222c44": "#d8e5f8",  # Profit highlight row bg (soft Latte Blue)
    "#292338": "#eadcf8",  # Linked / Mauve stage row bg (soft Latte Mauve)
    "#24273a": "#dce6fa",  # Blue stage row bg (soft Latte Blue)
    "#212e28": "#dcf0df",  # Green stage row bg (soft Latte Green)
    "#302623": "#fae4d4",  # Peach stage row bg (soft Latte Peach)
    "#30212b": "#f8dce4",  # Red stage row bg (soft Latte Red)
    "#1f2e30": "#daf0f0",  # Teal stage row bg (soft Latte Teal)
    "#291e28": "#f7e2ea",  # Gantt M5/M13 column bg
    "#452738": "#eecad8",  # Gantt M5/M13 month header bg
    "#1e292e": "#dff0f4",  # Gantt M6/M15 column bg
    "#253b3f": "#cae4eb",  # Gantt M6/M15 month header bg
    "#1e2926": "#dff2e4",  # Gantt M14 column bg
    "#233b30": "#c8e8d2",  # Gantt M14 month header bg
    # Catppuccin Accents
    "#b4befe": "#7287fd",  # Lavender
    "#89b4fa": "#1e66f5",  # Blue
    "#74c7ec": "#209fb5",  # Sapphire
    "#89dceb": "#04a5e5",  # Sky
    "#94e2d5": "#179299",  # Teal
    "#a6e3a1": "#40a02b",  # Green
    "#f9e2af": "#df8e1d",  # Yellow
    "#fab387": "#fe640b",  # Peach
    "#eba0ac": "#e64553",  # Maroon
    "#f38ba8": "#d20f39",  # Red
    "#cba6f7": "#8839ef",  # Mauve
}

# 2. Edge / Line / Spine / Border colors
EDGE_MOCHA_TO_LATTE: dict[str, str] = {
    **FACE_MOCHA_TO_LATTE,
    # Bar/marker/callout dark borders in Mocha (#11111b) become crisp Latte charcoal (#4c4f69)
    "#11111b": "#4c4f69",
    "#181825": "#eff1f5",
    "#313244": "#ccd0da",
    "#45475a": "#bcc0cc",
}

# 3. Text colors
TEXT_MOCHA_TO_LATTE: dict[str, str] = {
    **FACE_MOCHA_TO_LATTE,
    "#cdd6f4": "#4c4f69",  # Primary text
    "#bac2de": "#5c5f77",  # Secondary text / axis labels
    "#a6adc8": "#6c6f85",  # Muted slate text
    "#9399b2": "#7c7f93",  # Overlay2 text
    "#7f849c": "#8c8fa1",  # Overlay1 text
    # Text explicitly drawn in #11111b or #f5f7ff in Mocha is inside dark banners/bars -> crisp white in Latte
    "#11111b": "#ffffff",
    "#f5f7ff": "#ffffff",
    "#dbeafe": "#dbeafe",
    "#181825": "#4c4f69",
}

MOCHA_RCPARAMS: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#45475a",
    "axes.linewidth": 0.9,
    "axes.labelcolor": "#bac2de",
    "xtick.color": "#bac2de",
    "ytick.color": "#bac2de",
    "text.color": "#cdd6f4",
    "grid.color": "#313244",
    "grid.linestyle": "--",
    "grid.alpha": 0.85,
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#181825",
    "savefig.facecolor": "#1e1e2e",
    "savefig.edgecolor": "#1e1e2e",
    "legend.facecolor": "#1e1e2e",
    "legend.edgecolor": "#45475a",
    "legend.labelcolor": "#cdd6f4",
}

LATTE_RCPARAMS: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#bcc0cc",
    "axes.linewidth": 0.9,
    "axes.labelcolor": "#5c5f77",
    "xtick.color": "#5c5f77",
    "ytick.color": "#5c5f77",
    "text.color": "#4c4f69",
    "grid.color": "#ccd0da",
    "grid.linestyle": "--",
    "grid.alpha": 0.85,
    "figure.facecolor": "#eff1f5",
    "axes.facecolor": "#e6e9ef",
    "savefig.facecolor": "#eff1f5",
    "savefig.edgecolor": "#eff1f5",
    "legend.facecolor": "#eff1f5",
    "legend.edgecolor": "#bcc0cc",
    "legend.labelcolor": "#4c4f69",
}


def _rgba_to_hex(color_val: object) -> tuple[str | None, float | None]:
    """Convert any Matplotlib color specification to a lowercase 6-digit hex string and alpha."""
    if color_val is None:
        return None, None
    if isinstance(color_val, str) and color_val.lower() == "none":
        return "none", 0.0
    try:
        rgba = mcolors.to_rgba(color_val)
        return mcolors.to_hex(rgba, keep_alpha=False).lower(), float(rgba[3])
    except Exception:
        return None, None


def _map_color(color_val: object, mapping: dict[str, str], preserve_alpha: bool = True) -> object:
    """Map a Matplotlib color specification using `mapping` while preserving its alpha channel."""
    hex_str, alpha = _rgba_to_hex(color_val)
    if hex_str is None or hex_str == "none":
        return color_val
    mapped_hex = mapping.get(hex_str)
    if mapped_hex is None:
        return color_val
    if preserve_alpha and alpha is not None and alpha < 0.999:
        rgb = mcolors.to_rgb(mapped_hex)
        return (rgb[0], rgb[1], rgb[2], alpha)
    return mapped_hex


def _map_rgba_array(arr: np.ndarray, mapping: dict[str, str]) -> np.ndarray:
    """Map an (N, 4) RGBA numpy array of colors from Mocha to Latte while preserving alpha."""
    if arr is None or len(arr) == 0:
        return arr
    out = np.array(arr, dtype=float, copy=True)
    if out.ndim == 1 and out.size == 4:
        hex_c = mcolors.to_hex(out[:3], keep_alpha=False).lower()
        if hex_c in mapping:
            out[:3] = mcolors.to_rgb(mapping[hex_c])
        return out
    if out.ndim == 2 and out.shape[1] == 4:
        for i in range(out.shape[0]):
            if out[i, 3] <= 0.0:
                continue
            hex_c = mcolors.to_hex(out[i, :3], keep_alpha=False).lower()
            if hex_c in mapping:
                out[i, :3] = mcolors.to_rgb(mapping[hex_c])
    return out


def _transform_patch_to_latte(patch: mpatches.Patch, is_table_or_banner: bool = False) -> None:
    """Transform a Matplotlib Patch (`facecolor` and `edgecolor`) to Catppuccin Latte."""
    if patch is None:
        return
    try:
        fc = patch.get_facecolor()
        fc_hex, fc_alpha = _rgba_to_hex(fc)
        if fc_hex and fc_hex != "none" and (fc_alpha is None or fc_alpha > 0.0):
            mapped_fc = FACE_MOCHA_TO_LATTE.get(fc_hex)
            if mapped_fc:
                rgb = mcolors.to_rgb(mapped_fc)
                patch.set_facecolor((rgb[0], rgb[1], rgb[2], fc_alpha if fc_alpha is not None else 1.0))
    except Exception:
        pass

    try:
        ec = patch.get_edgecolor()
        ec_hex, ec_alpha = _rgba_to_hex(ec)
        if ec_hex and ec_hex != "none" and (ec_alpha is None or ec_alpha > 0.0):
            edge_map = FACE_MOCHA_TO_LATTE if is_table_or_banner else EDGE_MOCHA_TO_LATTE
            mapped_ec = edge_map.get(ec_hex)
            if mapped_ec:
                rgb = mcolors.to_rgb(mapped_ec)
                patch.set_edgecolor((rgb[0], rgb[1], rgb[2], ec_alpha if ec_alpha is not None else 1.0))
    except Exception:
        pass


def apply_latte_theme_to_figure(fig: mfigure.Figure) -> None:
    """Walk all Matplotlib artists in `fig` and convert Catppuccin Mocha colors to Catppuccin Latte."""
    # 1. Figure background patch
    _transform_patch_to_latte(fig.patch, is_table_or_banner=True)

    # 2. Figure-level texts (suptitle, subtitles, footers)
    for txt in fig.texts:
        c_hex, _ = _rgba_to_hex(txt.get_color())
        if c_hex in TEXT_MOCHA_TO_LATTE:
            txt.set_color(TEXT_MOCHA_TO_LATTE[c_hex])
        bbox = txt.get_bbox_patch()
        if bbox is not None:
            _transform_patch_to_latte(bbox)

    # 3. Walk every Axes in Figure
    for ax in fig.axes:
        _transform_patch_to_latte(ax.patch, is_table_or_banner=True)

        # Spines
        for spine in ax.spines.values():
            s_hex, s_alpha = _rgba_to_hex(spine.get_edgecolor())
            if s_hex in EDGE_MOCHA_TO_LATTE:
                rgb = mcolors.to_rgb(EDGE_MOCHA_TO_LATTE[s_hex])
                spine.set_edgecolor((rgb[0], rgb[1], rgb[2], s_alpha if s_alpha is not None else 1.0))

        # Axis labels, titles, ticks, and gridlines
        ax.title.set_color(_map_color(ax.title.get_color(), TEXT_MOCHA_TO_LATTE, preserve_alpha=False))
        ax.xaxis.label.set_color(_map_color(ax.xaxis.label.get_color(), TEXT_MOCHA_TO_LATTE, preserve_alpha=False))
        ax.yaxis.label.set_color(_map_color(ax.yaxis.label.get_color(), TEXT_MOCHA_TO_LATTE, preserve_alpha=False))

        for tick in list(ax.xaxis.get_major_ticks()) + list(ax.xaxis.get_minor_ticks()) + list(ax.yaxis.get_major_ticks()) + list(ax.yaxis.get_minor_ticks()):
            tick.label1.set_color(_map_color(tick.label1.get_color(), TEXT_MOCHA_TO_LATTE, preserve_alpha=False))
            tick.label2.set_color(_map_color(tick.label2.get_color(), TEXT_MOCHA_TO_LATTE, preserve_alpha=False))
            tick.tick1line.set_color(_map_color(tick.tick1line.get_color(), EDGE_MOCHA_TO_LATTE, preserve_alpha=False))
            tick.tick2line.set_color(_map_color(tick.tick2line.get_color(), EDGE_MOCHA_TO_LATTE, preserve_alpha=False))
            tick.gridline.set_color(_map_color(tick.gridline.get_color(), EDGE_MOCHA_TO_LATTE))

        # Check if this Axes hosts a heatmap image (e.g. AdaptivePricing 2-row hourly calendar schedule)
        has_heatmap_img = len(ax.images) > 0
        if has_heatmap_img:
            latte_heatmap_cmap = mcolors.LinearSegmentedColormap.from_list(
                "pick_price_cmap_latte",
                ["#9ce5db", "#9ee0f5", "#b0cefc", "#fae29c", "#fcc292", "#f7a6b9"],
            )
            for im in ax.images:
                im.set_cmap(latte_heatmap_cmap)
                if getattr(im, "colorbar", None) is not None:
                    try:
                        im.colorbar.update_normal(im)
                    except Exception:
                        pass

        # Patches (Bars, Rectangles, FancyBboxPatches, Polygons)
        for p in ax.patches:
            _transform_patch_to_latte(p)

        # Lines (Line2D)
        for line in ax.lines:
            line.set_color(_map_color(line.get_color(), FACE_MOCHA_TO_LATTE))
            line.set_markerfacecolor(_map_color(line.get_markerfacecolor(), FACE_MOCHA_TO_LATTE))
            line.set_markeredgecolor(_map_color(line.get_markeredgecolor(), EDGE_MOCHA_TO_LATTE))

        # Collections (Scatter PathCollections, fill_between PolyCollections, Errorbar LineCollections)
        for coll in ax.collections:
            if isinstance(coll, Collection):
                try:
                    cmap_obj = coll.get_cmap()
                    if cmap_obj is not None and getattr(cmap_obj, "name", "") == "pick_price_cmap":
                        coll.set_cmap(
                            mcolors.LinearSegmentedColormap.from_list(
                                "pick_price_cmap_latte",
                                ["#9ce5db", "#9ee0f5", "#b0cefc", "#fae29c", "#fcc292", "#f7a6b9"],
                            )
                        )
                except Exception:
                    pass
                try:
                    coll.set_facecolor(_map_rgba_array(coll.get_facecolor(), FACE_MOCHA_TO_LATTE))
                except Exception:
                    pass
                try:
                    coll.set_edgecolor(_map_rgba_array(coll.get_edgecolor(), EDGE_MOCHA_TO_LATTE))
                except Exception:
                    pass

        # Texts & Annotations on Axes
        for txt in ax.texts:
            c_hex, _ = _rgba_to_hex(txt.get_color())
            if has_heatmap_img and c_hex in ("#11111b", "#181825"):
                # Inside light-pastel heatmap cells, keep dark Latte charcoal text
                txt.set_color("#4c4f69" if c_hex == "#11111b" else "#5c5f77")
            elif c_hex in TEXT_MOCHA_TO_LATTE:
                txt.set_color(TEXT_MOCHA_TO_LATTE[c_hex])

            bbox = txt.get_bbox_patch()
            if bbox is not None:
                _transform_patch_to_latte(bbox)

        # Legend
        leg = ax.get_legend()
        if leg is not None:
            _transform_patch_to_latte(leg.get_frame(), is_table_or_banner=True)
            if leg.get_title() is not None:
                leg.get_title().set_color(_map_color(leg.get_title().get_color(), TEXT_MOCHA_TO_LATTE, preserve_alpha=False))
            for ltxt in leg.get_texts():
                ltxt.set_color(_map_color(ltxt.get_color(), TEXT_MOCHA_TO_LATTE, preserve_alpha=False))
            for lpatch in leg.get_patches():
                _transform_patch_to_latte(lpatch)
            for lline in leg.get_lines():
                lline.set_color(_map_color(lline.get_color(), FACE_MOCHA_TO_LATTE))
                lline.set_markerfacecolor(_map_color(lline.get_markerfacecolor(), FACE_MOCHA_TO_LATTE))
                lline.set_markeredgecolor(_map_color(lline.get_markeredgecolor(), EDGE_MOCHA_TO_LATTE))


def _themed_savefig(self: mfigure.Figure, *args, **kwargs):
    """
    Wrapper around `Figure.savefig` that applies the active Catppuccin theme (`Mocha` or `Latte`),
    renders the content area tightly, and attaches a standardized, pixel-identical Header Section
    across the top of every saved image.
    """
    import io
    import matplotlib.font_manager as fm
    from PIL import Image, ImageDraw, ImageFont

    req_dpi = kwargs.get("dpi")
    base_dpi = float(req_dpi) if isinstance(req_dpi, (int, float)) else float(self.dpi or 320.0)
    render_dpi = max(base_dpi, 320.0)

    # 1. Extract main chart title from `fig._suptitle` or top-center figure text and remove from content render
    header_title = ""
    if getattr(self, "_suptitle", None) is not None:
        header_title = self._suptitle.get_text().strip()
        if self._suptitle in self.texts:
            self.texts.remove(self._suptitle)
        self._suptitle = None

    remaining_texts = []
    for txt in list(self.texts):
        pos = txt.get_position()
        if len(pos) >= 2 and float(pos[1]) >= 0.935 and txt.get_ha() == "center":
            if not header_title:
                header_title = txt.get_text().strip()
        else:
            remaining_texts.append(txt)
    self.texts = remaining_texts

    if not header_title and len(self.axes) == 1:
        ax0_title = self.axes[0].get_title().strip()
        if ax0_title:
            header_title = ax0_title
            self.axes[0].set_title("")

    header_title = " ".join(header_title.split())

    # 2. Apply active Catppuccin theme (`Latte` or `Mocha`)
    active = get_theme()
    if active == "Latte":
        apply_latte_theme_to_figure(self)
        bg_hex = "#eff1f5"
        hdr_bg_rgb = (230, 233, 239)      # #e6e9ef (Latte Mantle)
        hdr_border_rgb = (188, 192, 204)  # #bcc0cc (Latte Surface1)
        hdr_text_rgb = (76, 79, 105)      # #4c4f69 (Latte Text)
        canvas_bg_rgb = (239, 241, 245)   # #eff1f5 (Latte Base)
    else:
        bg_hex = "#1e1e2e"
        hdr_bg_rgb = (24, 24, 37)         # #181825 (Mocha Mantle)
        hdr_border_rgb = (69, 71, 90)     # #45475a (Mocha Surface1)
        hdr_text_rgb = (205, 214, 244)    # #cdd6f4 (Mocha Text)
        canvas_bg_rgb = (30, 30, 46)      # #1e1e2e (Mocha Base)

    # 3. Render the content-only figure into an in-memory PNG buffer
    buf = io.BytesIO()
    save_kwargs = dict(kwargs)
    save_kwargs["dpi"] = render_dpi
    save_kwargs["format"] = "png"
    save_kwargs["bbox_inches"] = "tight"
    save_kwargs["pad_inches"] = 0.03
    save_kwargs["facecolor"] = bg_hex
    save_kwargs["edgecolor"] = bg_hex
    _ORIG_SAVEFIG(self, buf, **save_kwargs)
    buf.seek(0)

    content_img = Image.open(buf).convert("RGB")
    arr = np.asarray(content_img, dtype=np.int16)
    bg_arr = np.array(canvas_bg_rgb, dtype=np.int16)
    mask = np.any(np.abs(arr - bg_arr) > 2, axis=2)
    if np.any(mask):
        ys, xs = np.where(mask)
        content_img = content_img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    # 4. Normalize content width and compose with the universal Header Section
    TARGET_W = 6400
    PAD_X = 64
    HEADER_H = 188 if header_title else 0
    BORDER_H = 4 if header_title else 0
    PAD_TOP = 38 if header_title else 48
    PAD_BOTTOM = 48

    content_target_w = TARGET_W - 2 * PAD_X
    scale = content_target_w / float(max(1, content_img.width))
    content_target_h = max(1, int(round(content_img.height * scale)))
    content_resized = content_img.resize((content_target_w, content_target_h), Image.Resampling.LANCZOS)

    total_h = HEADER_H + PAD_TOP + content_target_h + PAD_BOTTOM
    final_img = Image.new("RGB", (TARGET_W, total_h), canvas_bg_rgb)
    draw = ImageDraw.Draw(final_img)

    if header_title:
        # Header section background and bottom divider rule
        draw.rectangle([0, 0, TARGET_W, HEADER_H - BORDER_H], fill=hdr_bg_rgb)
        draw.rectangle([0, HEADER_H - BORDER_H, TARGET_W, HEADER_H], fill=hdr_border_rgb)

        ttf_path = fm.findfont(fm.FontProperties(family="sans-serif", weight="bold"))
        font_sz = 82
        font = ImageFont.truetype(ttf_path, size=font_sz)
        bbox = draw.textbbox((TARGET_W // 2, (HEADER_H - BORDER_H) // 2), header_title, font=font, anchor="mm")
        while (bbox[2] - bbox[0]) > (TARGET_W - 2 * PAD_X) and font_sz > 48:
            font_sz -= 2
            font = ImageFont.truetype(ttf_path, size=font_sz)
            bbox = draw.textbbox((TARGET_W // 2, (HEADER_H - BORDER_H) // 2), header_title, font=font, anchor="mm")

        draw.text(
            (TARGET_W // 2, (HEADER_H - BORDER_H) // 2),
            header_title,
            font=font,
            fill=hdr_text_rgb,
            anchor="mm",
        )

    final_img.paste(content_resized, (PAD_X, HEADER_H + PAD_TOP))
    out_target = args[0] if args else kwargs.get("fname")
    final_img.save(out_target, format="PNG", dpi=(320, 320))
    return None


def install_theme_hook() -> None:
    """Install the Matplotlib `Figure.savefig` theme hook once per process."""
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        mfigure.Figure.savefig = _themed_savefig  # type: ignore[method-assign]
        _HOOK_INSTALLED = True


def set_theme(theme_name: str | None = None) -> str:
    """
    Set the active Catppuccin color scheme (`Mocha` dark mode or `Latte` light mode),
    update `plt.rcParams`, and install the `Figure.savefig` hook.
    """
    global _ACTIVE_THEME
    if theme_name is not None:
        norm_name = theme_name.strip().lower()
        if norm_name in ("latte", "light"):
            _ACTIVE_THEME = "Latte"
        elif norm_name in ("mocha", "dark"):
            _ACTIVE_THEME = "Mocha"
        else:
            raise ValueError(f"Unsupported theme '{theme_name}'. Choose 'Mocha' (dark) or 'Latte' (light).")
    os.environ["PICK_THEME"] = _ACTIVE_THEME
    install_theme_hook()
    plt.rcParams.update(LATTE_RCPARAMS if _ACTIVE_THEME == "Latte" else MOCHA_RCPARAMS)
    return _ACTIVE_THEME


def get_theme() -> str:
    """Return the active Catppuccin theme name (`'Mocha'` or `'Latte'`)."""
    env_theme = os.environ.get("PICK_THEME", _ACTIVE_THEME).strip().capitalize()
    return "Latte" if env_theme == "Latte" else "Mocha"


# Install hook on import so any module importing `theme` is automatically theme-aware
install_theme_hook()
