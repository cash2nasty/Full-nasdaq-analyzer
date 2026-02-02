import pandas as pd


def style_stats_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def bias_color(val):
        if val == "Bullish":
            return "background-color: #c6f6d5"  # green
        if val == "Bearish":
            return "background-color: #fed7d7"  # red
        if val == "Neutral":
            return "background-color: #fefcbf"  # yellow
        return ""

    def overext_color(val):
        if "Overextended" in str(val):
            return "background-color: #bee3f8"  # blue
        return ""

    def rvol_color(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v > 1.5:
            return "background-color: #fbd38d"  # orange
        if v < 0.7:
            return "background-color: #e2e8f0"  # gray
        return ""

    def vol_color(val):
        return ""

    styler = df.style

    def _apply_elementwise(styler_obj, func, cols):
        # If Styler has applymap (older pandas), use it. Otherwise use apply
        if hasattr(styler_obj, "applymap"):
            return styler_obj.applymap(func, subset=cols)
        # apply expects a function returning a Series of CSS strings for the column
        for c in cols:
            if c in df.columns:
                styler_obj = styler_obj.apply(lambda s: s.map(func), subset=[c])
        return styler_obj

    if "Daily_Bias" in df.columns:
        styler = _apply_elementwise(styler, bias_color, ["Daily_Bias"])
    if "Overextension" in df.columns:
        styler = _apply_elementwise(styler, overext_color, ["Overextension"])
    if "RVOL" in df.columns:
        styler = _apply_elementwise(styler, rvol_color, ["RVOL"])
    if "Volatility" in df.columns:
        styler = _apply_elementwise(styler, vol_color, ["Volatility"])
    return styler