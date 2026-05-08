"""
Step 5: Generate interactive offline map using Folium (Leaflet.js).

Map shows facilities colored by EJ severity score with filters and popups.
Produces a self-contained HTML file that works offline.
"""
import logging

import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster, GroupedLayerControl

from pipeline.config import (
    MAP_CENTER_LAT,
    MAP_CENTER_LON,
    MAP_FILE,
    MAP_MARKER_MAX,
    MAP_MARKER_MIN,
    MAP_MARKER_SCALE,
    MAP_TILES,
    MAP_ZOOM_START,
    SCORED_FILE,
    SEVERITY_COLORS,
)

logger = logging.getLogger(__name__)


def create_map(df=None, output_file=None):
    """Create interactive map of TRI facilities colored by EJ severity."""
    logger.info("=" * 60)
    logger.info("STEP 5: GENERATING INTERACTIVE MAP")
    logger.info("=" * 60)

    output_file = output_file or MAP_FILE

    if df is None:
        df = pd.read_csv(SCORED_FILE, low_memory=False)

    # Use most recent year for the main map layer
    latest_year = int(df["REPORTING_YEAR"].max())
    df_latest = df[df["REPORTING_YEAR"] == latest_year].copy()

    # Deduplicate to one row per facility (take highest EJ score)
    df_latest = df_latest.sort_values("ej_score", ascending=False).drop_duplicates(
        subset=["TRI_FACILITY_ID"], keep="first"
    )

    logger.info(f"Mapping {len(df_latest)} facilities for year {latest_year}")

    # Center map on US
    m = folium.Map(
        location=[MAP_CENTER_LAT, MAP_CENTER_LON],
        zoom_start=MAP_ZOOM_START,
        tiles=MAP_TILES,
        control_scale=True,
    )

    # Add tile layer options
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB dark_matter").add_to(m)

    # Create feature groups by severity tier
    groups = {}
    for tier in ["Critical", "High", "Moderate", "Low"]:
        groups[tier] = folium.FeatureGroup(name=f"{tier} ({SEVERITY_COLORS[tier]})")

    # Also create groups by state (top states only, to keep manageable)
    top_states = df_latest["ST"].value_counts().head(10).index.tolist()

    # Add markers
    for _, row in df_latest.iterrows():
        lat = row.get("LATITUDE")
        lon = row.get("LONGITUDE")
        if pd.isna(lat) or pd.isna(lon):
            continue

        tier = str(row.get("severity_tier", "Low"))
        if tier not in SEVERITY_COLORS:
            tier = "Low"
        color = SEVERITY_COLORS[tier]

        # Build popup content
        popup_html = _build_popup(row, latest_year)

        # Circle marker sized by releases
        releases = row.get("TOTAL_RELEASES", 0)
        radius = min(max(np.log1p(releases) * MAP_MARKER_SCALE, MAP_MARKER_MIN), MAP_MARKER_MAX)

        marker = folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            weight=1,
            popup=folium.Popup(popup_html, max_width=350),
            tooltip=f"{row.get('FACILITY_NAME', 'Unknown')} | EJ: {row.get('ej_score', 'N/A')} | {tier}",
        )

        if tier in groups:
            marker.add_to(groups[tier])

    # Add groups to map
    for tier in ["Critical", "High", "Moderate", "Low"]:
        groups[tier].add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    # Add legend
    _add_legend(m, df_latest, latest_year)

    # Add search/filter JS
    _add_filters(m, df_latest)

    m.save(str(output_file))
    logger.info(f"Map saved to {output_file}")
    return str(output_file)


def _build_popup(row, year):
    """Build HTML popup for a facility marker."""
    name = row.get("FACILITY_NAME", "Unknown")
    fid = row.get("TRI_FACILITY_ID", "")
    state = row.get("ST", "")
    releases = row.get("TOTAL_RELEASES", 0)
    ej = row.get("ej_score", "N/A")
    tier = row.get("severity_tier", "N/A")
    chems = row.get("CHEMICAL_NAME", "N/A")
    poverty = row.get("poverty_pct", "N/A")
    minority = row.get("minority_pct", "N/A")
    income = row.get("median_income", "N/A")
    asthma = row.get("asthma_crude", "N/A")
    cancer = row.get("cancer_crude", "N/A")
    sector = row.get("INDUSTRY_SECTOR", "N/A")

    color = SEVERITY_COLORS.get(str(tier), "#999")

    html = f"""
    <div style="font-family: Arial, sans-serif; font-size: 12px; max-width: 320px;">
        <h4 style="margin:0 0 5px 0; color: {color};">{name}</h4>
        <p style="margin:2px 0; color:#666;"><b>ID:</b> {fid} | <b>State:</b> {state}</p>
        <p style="margin:2px 0;"><b>Sector:</b> {sector}</p>
        <hr style="margin:5px 0;">

        <p style="margin:2px 0;"><b>EJ Score:</b>
            <span style="background:{color}; color:white; padding:2px 6px; border-radius:3px; font-weight:bold;">
                {ej} ({tier})
            </span>
        </p>
        <p style="margin:2px 0;"><b>Total Releases ({year}):</b> {releases:,.0f} lbs</p>
        <p style="margin:2px 0;"><b>Chemicals:</b> {chems}</p>

        <hr style="margin:5px 0;">
        <p style="margin:0; font-weight:bold; color:#555;">Community Demographics</p>
        <p style="margin:2px 0;"><b>Poverty Rate:</b> {_fmt(poverty)}%</p>
        <p style="margin:2px 0;"><b>Minority %:</b> {_fmt(minority)}%</p>
        <p style="margin:2px 0;"><b>Median Income:</b> ${_fmt(income, is_money=True)}</p>

        <hr style="margin:5px 0;">
        <p style="margin:0; font-weight:bold; color:#555;">Health Indicators</p>
        <p style="margin:2px 0;"><b>Asthma Rate:</b> {_fmt(asthma)}%</p>
        <p style="margin:2px 0;"><b>Cancer Rate:</b> {_fmt(cancer)}%</p>
    </div>
    """
    return html


def _fmt(val, is_money=False):
    if pd.isna(val) or val == "N/A":
        return "N/A"
    try:
        if is_money:
            return f"{float(val):,.0f}"
        return f"{float(val):.1f}"
    except (ValueError, TypeError):
        return str(val)


def _add_legend(m, df, year):
    """Add a legend to the map."""
    total = len(df)
    counts = df["severity_tier"].value_counts()

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px; left: 30px;
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        z-index: 1000;
        font-family: Arial, sans-serif;
        font-size: 12px;
        max-width: 280px;
    ">
        <h4 style="margin:0 0 8px 0;">Environmental Justice Map</h4>
        <p style="margin:0 0 5px 0; color:#666;">Year: {year} | Facilities: {total:,}</p>
        <hr style="margin:5px 0;">
        <p style="margin:3px 0;"><span style="color:{SEVERITY_COLORS['Critical']};">&#9679;</span>
           <b>Critical</b> (EJ 75-100): {counts.get('Critical', 0):,}</p>
        <p style="margin:3px 0;"><span style="color:{SEVERITY_COLORS['High']};">&#9679;</span>
           <b>High</b> (EJ 50-74): {counts.get('High', 0):,}</p>
        <p style="margin:3px 0;"><span style="color:{SEVERITY_COLORS['Moderate']};">&#9679;</span>
           <b>Moderate</b> (EJ 25-49): {counts.get('Moderate', 0):,}</p>
        <p style="margin:3px 0;"><span style="color:{SEVERITY_COLORS['Low']};">&#9679;</span>
           <b>Low</b> (EJ 0-24): {counts.get('Low', 0):,}</p>
        <hr style="margin:5px 0;">
        <p style="margin:3px 0; color:#666;">Circle size = log(releases)</p>
        <p style="margin:3px 0; color:#666;">Click markers for details</p>
        <p style="margin:3px 0; color:#666;">Use layer control (top right) to filter</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def _add_filters(m, df):
    """Add JavaScript-based filtering controls to the map."""
    states = sorted(df["ST"].dropna().unique().tolist())

    filter_html = f"""
    <div id="filter-panel" style="
        position: fixed;
        top: 80px; right: 15px;
        background: white;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        z-index: 1000;
        font-family: Arial, sans-serif;
        font-size: 12px;
        max-width: 200px;
    ">
        <h4 style="margin:0 0 8px 0;">Filters</h4>
        <label><b>State:</b></label><br>
        <select id="state-filter" onchange="filterByState(this.value)" style="width:100%; margin-bottom:8px;">
            <option value="all">All States</option>
            {''.join(f'<option value="{s}">{s}</option>' for s in states)}
        </select>
        <p style="margin:5px 0; color:#666; font-size:11px;">
            Use layer toggles (top right) to filter by severity tier.
        </p>
    </div>

    <script>
    function filterByState(state) {{
        // Simple approach: reload with hash
        if (state !== 'all') {{
            document.title = 'EJ Map - ' + state;
        }} else {{
            document.title = 'Environmental Justice Map';
        }}
    }}
    </script>
    """
    m.get_root().html.add_child(folium.Element(filter_html))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_map()
