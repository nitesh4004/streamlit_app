import streamlit as st
import ee
import json
import geemap.foliumap as geemap
import xml.etree.ElementTree as ET
import re
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta
import numpy as np

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="NI30 Orbital Analytics", 
    page_icon="🛰️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED CSS STYLING (Cyber-Glass UI) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&family=Inter:wght@300;400;600&display=swap');
    
    /* GLOBAL VARIABLES */
    :root {
        --bg-color: #050509;
        --card-bg: rgba(20, 24, 35, 0.7);
        --glass-border: 1px solid rgba(255, 255, 255, 0.08);
        --accent-primary: #00f2ff;
        --accent-secondary: #7000ff;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
    }

    /* BACKGROUND */
    .stApp { 
        background-image: radial-gradient(circle at 50% 0%, #1a1f35 0%, #050509 100%);
        font-family: 'Inter', sans-serif;
    }

    /* TYPOGRAPHY */
    h1, h2, h3, .title-font { font-family: 'Rajdhani', sans-serif !important; text-transform: uppercase; letter-spacing: 1px; }
    p, label, .stMarkdown, div { color: var(--text-primary) !important; }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 12, 16, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    /* WIDGET STYLING */
    .stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div, .stDateInput > div > div {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 4px;
        color: #fff !important;
        transition: all 0.3s ease;
    }
    .stTextInput > div > div:focus-within { border-color: var(--accent-primary) !important; box-shadow: 0 0 10px rgba(0, 242, 255, 0.2); }

    /* BUTTONS */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, var(--accent-secondary) 0%, #4c1d95 100%);
        border: none;
        color: white;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.6rem;
        border-radius: 4px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(112, 0, 255, 0.4);
    }
    div.stButton > button:first-child:active { transform: translateY(0); }

    /* CUSTOM HUD HEADER */
    .hud-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(255, 255, 255, 0.02);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding: 15px 25px;
        border-radius: 0 0 15px 15px;
        margin-bottom: 25px;
    }
    .hud-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        background: -webkit-linear-gradient(0deg, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hud-badge {
        background: rgba(0, 242, 255, 0.1);
        border: 1px solid rgba(0, 242, 255, 0.3);
        color: var(--accent-primary);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }

    /* GLASS CARDS */
    .glass-card {
        background: var(--card-bg);
        border: var(--glass-border);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .card-label {
        font-family: 'Rajdhani', sans-serif;
        color: var(--accent-primary);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 5px;
    }

    /* EXPANDER STYLE */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.02) !important;
        border-radius: 5px;
    }
    
    /* MAP CONTAINER */
    iframe {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AUTHENTICATION ---
try:
    service_account = st.secrets["gcp_service_account"]["client_email"]
    secret_dict = dict(st.secrets["gcp_service_account"])
    key_data = json.dumps(secret_dict) 
    credentials = ee.ServiceAccountCredentials(service_account, key_data=key_data)
    ee.Initialize(credentials)
except Exception as e:
    try:
        ee.Initialize()
    except Exception as e_inner:
        st.error(f"⚠️ Authentication Error: {e}")
        st.stop()

if 'calculated' not in st.session_state: st.session_state['calculated'] = False
if 'dates' not in st.session_state: st.session_state['dates'] = []
if 'roi' not in st.session_state: st.session_state['roi'] = None

# --- 4. FUNCTIONS (LOGIC PRESERVED) ---
def parse_kml(content):
    try:
        if isinstance(content, bytes): content = content.decode('utf-8')
        match = re.search(r'<coordinates>(.*?)</coordinates>', content, re.DOTALL | re.IGNORECASE)
        if match: return process_coords(match.group(1))
        root = ET.fromstring(content)
        for elem in root.iter():
            if elem.tag.lower().endswith('coordinates') and elem.text:
                return process_coords(elem.text)
    except: pass
    return None

def process_coords(text):
    raw = text.strip().split()
    coords = [[float(x.split(',')[0]), float(x.split(',')[1])] for x in raw if len(x.split(',')) >= 2]
    return ee.Geometry.Polygon([coords]) if len(coords) > 2 else None

def preprocess_landsat(img):
    opticalBands = img.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBands = img.select('ST_B.*').multiply(0.00341802).add(149.0)
    return img.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

def rename_landsat_bands(img):
    return img.select(
        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    )

def compute_index(img, platform, index, formula=None):
    if platform == "Sentinel-2 (Optical)":
        if index == '🛠️ Custom (Band Math)':
            map_b = {
                'B1':img.select('B1'), 'B2':img.select('B2'), 'B3':img.select('B3'), 'B4':img.select('B4'), 
                'B5':img.select('B5'), 'B6':img.select('B6'), 'B7':img.select('B7'),
                'B8':img.select('B8'), 'B8A':img.select('B8A'), 
                'B11':img.select('B11'), 'B12':img.select('B12')
            }
            return img.expression(formula, map_b).rename('Custom')
        map_i = {'NDVI': ['B8','B4'], 'GNDVI': ['B8','B3'], 'NDWI (Water)': ['B3','B8'], 'NDMI': ['B8','B11']}
        if index in map_i: return img.normalizedDifference(map_i[index]).rename(index.split()[0])

    elif "Landsat" in platform:
        if index == '🛠️ Custom (Band Math)':
            map_b = {'B1':img.select('B1'), 'B2':img.select('B2'), 'B3':img.select('B3'), 'B4':img.select('B4'), 'B5':img.select('B5'), 'B6':img.select('B6'), 'B7':img.select('B7')}
            return img.expression(formula, map_b).rename('Custom')
        map_i = {'NDVI': ['B5','B4'], 'GNDVI': ['B5','B3'], 'NDWI (Water)': ['B3','B5'], 'NDMI': ['B5','B6']}
        if index in map_i: return img.normalizedDifference(map_i[index]).rename(index.split()[0])

    elif platform == "Sentinel-1 (Radar)":
        if index == '🛠️ Custom (Band Math)':
            map_b = {'VV': img.select('VV'), 'VH': img.select('VH')}
            return img.expression(formula, map_b).rename('Custom')
        if index == 'VV': return img.select('VV')
        if index == 'VH': return img.select('VH')
        if index == 'VH/VV Ratio': return img.select('VH').subtract(img.select('VV')).rename('Ratio')
    return img.select(0)

def generate_static_map_display(image, roi, vis_params, title, cmap_colors):
    # (Function logic preserved, styling adjusted for plots)
    thumb_url = image.getThumbURL({
        'min': vis_params['min'], 'max': vis_params['max'],
        'palette': vis_params['palette'], 'region': roi,
        'dimensions': 600, 'format': 'png'
    })
    response = requests.get(thumb_url)
    img_pil = Image.open(BytesIO(response.content))
    
    coords = roi.bounds().getInfo()['coordinates'][0]
    min_lon, min_lat = min(c[0] for c in coords), min(c[1] for c in coords)
    max_lon, max_lat = max(c[0] for c in coords), max(c[1] for c in coords)
    center_lat = (min_lat + max_lat) / 2
    deg_to_m = 111320 * np.cos(np.radians(center_lat))
    width_m = (max_lon - min_lon) * deg_to_m
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=80, facecolor='#050509')
    ax.set_facecolor('#050509')
    ax.imshow(img_pil)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color='#00f2ff', fontname='Sans Serif')
    
    img_w_px = img_pil.width
    scale_bar_m = width_m * 0.2
    scale_text = f"{scale_bar_m/1000:.1f} km" if scale_bar_m > 1000 else f"{int(scale_bar_m)} m"
    bar_x, bar_y = img_w_px * 0.05, img_pil.height * 0.95
    ax.add_patch(Rectangle((bar_x, bar_y - 5), img_w_px * 0.2, 10, color='white'))
    ax.text(bar_x + (img_w_px * 0.1), bar_y - 15, scale_text, ha='center', color='white', fontsize=10, weight='bold')

    cmap = mcolors.LinearSegmentedColormap.from_list("custom", cmap_colors)
    norm = mcolors.Normalize(vmin=vis_params['min'], vmax=vis_params['max'])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Index Value', fontsize=10, color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    buf = BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight', facecolor='#050509')
    buf.seek(0)
    plt.close(fig)
    return buf

# --- 5. SIDEBAR (CONTROL PANEL) ---
with st.sidebar:
    st.markdown("""
        <div style="margin-bottom: 20px;">
            <h2 style="font-family: 'Rajdhani'; color: #fff; margin:0;">NI30</h2>
            <p style="font-size: 0.8rem; color: #00f2ff; letter-spacing: 2px; margin:0;">GEOSPATIAL CORE</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("### 1. Target Acquisition (ROI)")
        roi_method = st.radio("Selection Mode", ["Upload KML", "Point & Buffer", "Manual Coordinates"], label_visibility="collapsed")
        
        new_roi = None
        if roi_method == "Upload KML":
            kml = st.file_uploader("Drop KML File", type=['kml'])
            if kml:
                kml.seek(0)
                new_roi = parse_kml(kml.read())
        elif roi_method == "Point & Buffer":
            c1, c2 = st.columns([1, 1])
            lat = c1.number_input("Lat", 20.59)
            lon = c2.number_input("Lon", 78.96)
            rad = st.number_input("Radius (km)", 5)
            if lat and lon: new_roi = ee.Geometry.Point([lon, lat]).buffer(rad*1000).bounds()
        elif roi_method == "Manual Coordinates":
            c1, c2 = st.columns(2)
            min_lon = c1.number_input("Min Lon", 78.0)
            min_lat = c2.number_input("Min Lat", 20.0)
            max_lon = c1.number_input("Max Lon", 79.0)
            max_lat = c2.number_input("Max Lat", 21.0)
            if min_lon < max_lon: new_roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

        if new_roi:
            if st.session_state['roi'] is None or new_roi.getInfo() != st.session_state['roi'].getInfo():
                st.session_state['roi'] = new_roi
                st.session_state['calculated'] = False
                st.session_state['dates'] = []
                st.toast("Target Locked: ROI Updated", icon="🎯")

    st.markdown("---")
    
    with st.container():
        st.markdown("### 2. Sensor Config")
        platform = st.selectbox("Satellite Network", [
            "Sentinel-2 (Optical)", "Landsat 9 (Optical)", "Landsat 8 (Optical)", "Sentinel-1 (Radar)"
        ])
        
        is_optical = "Optical" in platform
        formula, vmin, vmax, orbit = "", 0, 1, "BOTH"
        
        if is_optical:
            idx = st.selectbox("Spectral Product", ['NDVI', 'GNDVI', 'NDWI (Water)', 'NDMI', '🛠️ Custom (Band Math)'])
            if 'Custom' in idx:
                def_form = "(B5-B4)/(B5+B4)" if "Landsat" in platform else "(B8-B4)/(B8+B4)"
                formula = st.text_input("Math Expression", def_form)
                pal_name = "Viridis"
            elif 'Water' in idx:
                vmin, vmax = -0.5, 0.5
                pal_name = "Blue-White-Green"
            else:
                vmin, vmax = 0.0, 0.8
                pal_name = "Red-Yellow-Green"
            
            c1, c2 = st.columns(2)
            vmin = c1.number_input("Min Thresh", value=vmin)
            vmax = c2.number_input("Max Thresh", value=vmax)
            cloud = st.slider("Cloud Tolerance %", 0, 30, 10)
        else:
            idx = st.selectbox("Polarization", ['VV', 'VH', 'VH/VV Ratio', '🛠️ Custom (Band Math)'])
            if 'Custom' in idx:
                formula = st.text_input("Expression", "VH/VV")
                pal_name = "Viridis"
            elif 'Ratio' in idx:
                vmin, vmax = -20.0, 0.0
                pal_name = "Magma"
            else:
                vmin, vmax = -25.0, -5.0
                pal_name = "Greyscale"
            
            c1, c2 = st.columns(2)
            vmin = c1.number_input("Min dB", value=vmin)
            vmax = c2.number_input("Max dB", value=vmax)
            orbit = st.radio("Pass Direction", ["DESCENDING", "ASCENDING", "BOTH"])
            cloud = 0

        pal_name = st.selectbox("Color Ramp", ["Red-Yellow-Green", "Blue-White-Green", "Magma", "Viridis", "Greyscale"], index=["Red-Yellow-Green", "Blue-White-Green", "Magma", "Viridis", "Greyscale"].index(pal_name))
        
    pal_map = {
        "Red-Yellow-Green": ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'],
        "Blue-White-Green": ['blue', 'white', 'green'],
        "Magma": ['black', 'purple', 'orange', 'white'],
        "Viridis": ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
        "Greyscale": ['black', 'white']
    }
    cur_palette = pal_map.get(pal_name, pal_map["Red-Yellow-Green"])

    st.markdown("---")
    st.markdown("### 3. Temporal Window")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", datetime.now()-timedelta(60))
    end = c2.date_input("End", datetime.now())

    st.markdown("###")
    if st.button("INITIALIZE SCAN 🚀"):
        if st.session_state['roi']:
            st.session_state.update({
                'calculated': True, 'platform': platform, 'idx': idx, 
                'start': start.strftime("%Y-%m-%d"), 'end': end.strftime("%Y-%m-%d"),
                'cloud': cloud, 'formula': formula, 'orbit': orbit,
                'vmin': vmin, 'vmax': vmax, 'palette': cur_palette
            })
            st.session_state['dates'] = []
        else:
            st.error("❌ Error: ROI not defined.")

# --- 6. MAIN CONTENT ---
# Custom HUD Header
st.markdown("""
<div class="hud-header">
    <div>
        <div class="hud-title">NI30 ANALYTICS</div>
        <div style="color:#94a3b8; font-size:0.9rem;">SATELLITE INTELLIGENCE DASHBOARD</div>
    </div>
    <div style="text-align:right;">
        <span class="hud-badge">SYSTEM ONLINE</span>
        <div style="font-family:'Rajdhani'; font-size:1.2rem; margin-top:5px;">""" + datetime.now().strftime("%H:%M UTC") + """</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- DASHBOARD LOGIC ---
if not st.session_state['calculated']:
    # Welcome / Landing View
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:40px;">
        <h2 style="color:#fff;">📡 WAITING FOR INPUT</h2>
        <p style="color:#94a3b8; margin-bottom:20px;">Configure the sensor parameters and region of interest in the sidebar panel.</p>
        <div style="display:flex; justify-content:center; gap:20px; flex-wrap:wrap;">
             <div style="background:rgba(255,255,255,0.05); padding:10px 20px; border-radius:5px;">1. Select Region</div>
             <div style="background:rgba(255,255,255,0.05); padding:10px 20px; border-radius:5px;">2. Choose Sensor</div>
             <div style="background:rgba(255,255,255,0.05); padding:10px 20px; border-radius:5px;">3. Initialize Scan</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Default Map
    m = geemap.Map(height=500, basemap="HYBRID")
    if st.session_state['roi']:
        m.centerObject(st.session_state['roi'], 12)
        m.addLayer(ee.Image().paint(st.session_state['roi'], 2, 3), {'palette': '#00f2ff'}, 'Target ROI')
    
    m.to_streamlit()

else:
    # --- PROCESSING & RESULTS ---
    roi = st.session_state['roi']
    p = st.session_state
    
    with st.spinner("🛰️ Establishing Uplink... Processing Earth Engine Data..."):
        # (Processing Logic Identical to original, just wrapped)
        if p['platform'] == "Sentinel-2 (Optical)":
            col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterBounds(roi).filterDate(p['start'], p['end'])
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', p['cloud'])))
            processed = col.map(lambda img: img.addBands(compute_index(img, p['platform'], p['idx'], p['formula'])))
        elif "Landsat" in p['platform']:
            col_raw = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") if "Landsat 9" in p['platform'] else ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            col = (col_raw.filterBounds(roi).filterDate(p['start'], p['end'])
                   .filter(ee.Filter.lt('CLOUD_COVER', p['cloud'])))
            def process_landsat_step(img):
                scaled = preprocess_landsat(img)
                renamed = rename_landsat_bands(scaled)
                return renamed.addBands(compute_index(renamed, p['platform'], p['idx'], p['formula']))
            processed = col.map(process_landsat_step)
        else:
            col = (ee.ImageCollection('COPERNICUS/S1_GRD')
                   .filterBounds(roi).filterDate(p['start'], p['end'])
                   .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')))
            if p['orbit'] != "BOTH": col = col.filter(ee.Filter.eq('orbitProperties_pass', p['orbit']))
            processed = col.map(lambda img: img.addBands(compute_index(img, p['platform'], p['idx'], p['formula'])))
        
        if not st.session_state['dates']:
            cnt = processed.size().getInfo()
            if cnt > 0:
                dates_list = processed.aggregate_array('system:time_start').map(
                    lambda t: ee.Date(t).format('YYYY-MM-dd')).distinct().sort()
                st.session_state['dates'] = dates_list.slice(0, 50).getInfo()
            else:
                st.error(f"⚠️ Signal Lost: No images found for parameters.")
                st.session_state['calculated'] = False

    if st.session_state['dates']:
        dates = st.session_state['dates']
        
        # --- UI LAYOUT FOR RESULTS ---
        col_map, col_data = st.columns([3, 1])
        
        with col_data:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">📅 ACQUISITION DATE</div>', unsafe_allow_html=True)
            sel_date = st.selectbox("Select Timestamp", dates, index=len(dates)-1, label_visibility="collapsed")
            st.caption(f"{len(dates)} Scenes Available")
            st.markdown('</div>', unsafe_allow_html=True)

            d_s = sel_date
            d_e = (datetime.strptime(sel_date, "%Y-%m-%d") + timedelta(1)).strftime("%Y-%m-%d")
            band = 'Custom' if 'Custom' in p['idx'] else p['idx'].split()[0]
            if 'Ratio' in p['idx']: band = 'Ratio'
            
            final_img = processed.filterDate(d_s, d_e).select(band).median().clip(roi)
            vis = {'min': p['vmin'], 'max': p['vmax'], 'palette': p['palette']}
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">💾 DATA EXPORT</div>', unsafe_allow_html=True)
            try:
                url = final_img.getDownloadURL({'scale': 30 if "Landsat" in p['platform'] else 10, 'region': roi, 'name': f"{band}_{sel_date}"})
                st.markdown(f"<a href='{url}' style='color:#00f2ff; text-decoration:none;'>🔗 Download GeoTIFF</a>", unsafe_allow_html=True)
            except: st.caption("Region too large for instant link.")
            
            st.markdown("---")
            if st.button("☁️ Save to Drive", use_container_width=True):
                ee.batch.Export.image.toDrive(image=final_img, description=f"{band}_{sel_date}", 
                                              scale=30 if "Landsat" in p['platform'] else 10, region=roi, folder='GEE_Exports').start()
                st.toast("Export Task Initiated")
            
            st.markdown("---")
            if st.button("📷 Render Map (JPG)", use_container_width=True):
                with st.spinner("Rendering..."):
                    buf = generate_static_map_display(final_img, roi, vis, f"{p['idx']} | {sel_date}", p['palette'])
                    st.download_button("⬇️ Save Image", buf, f"Ni30_Map_{sel_date}.jpg", "image/jpeg", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_map:
            m = geemap.Map(height=700, basemap="HYBRID")
            m.centerObject(roi, 13)
            m.addLayer(final_img, vis, f"{p['idx']} ({sel_date})")
            m.add_colorbar(vis, label=p['idx'], layer_name="Legend")
            m.to_streamlit()
