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
    page_title="Geospatial Ni30", 
    page_icon="🌍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #0f172a; }
    section[data-testid="stSidebar"] * { color: #f8fafc !important; }
    section[data-testid="stSidebar"] .stTextInput > div > div,
    section[data-testid="stSidebar"] .stNumberInput > div > div,
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stDateInput > div > div {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
    }
    section[data-testid="stSidebar"] svg { fill: #ffffff !important; }
    .control-card {
        background: white; padding: 15px; border-radius: 8px;
        border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 15px;
    }
    .card-header {
        font-size: 0.75rem; text-transform: uppercase; font-weight: 700;
        color: #64748b; margin-bottom: 8px; border-bottom: 1px solid #f1f5f9; padding-bottom: 5px;
    }
    .navbar {
        background: white; padding: 0.8rem 1.5rem; border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px;
        border: 1px solid #e2e8f0; display: flex; justify-content: space-between;
    }
    .navbar-title { font-size: 1.2rem; font-weight: 700; color: #0f172a; }
    div.stButton > button:first-child {
        background-color: #2563eb; color: white; border-radius: 6px; border: none;
        padding: 0.5rem; font-weight: 600; width: 100%;
    }
    div.stButton > button:first-child:hover { background-color: #1d4ed8; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AUTHENTICATION (FIXED) ---
try:
    # 1. Fetch the secrets
    service_account = st.secrets["gcp_service_account"]["client_email"]
    
    # 2. THE FIX: Convert the secret object back into a JSON string
    # Streamlit gives us an 'AttrDict', but Earth Engine wants a 'String'
    secret_dict = dict(st.secrets["gcp_service_account"])
    key_data = json.dumps(secret_dict) 
    
    # 3. Authenticate using the string
    credentials = ee.ServiceAccountCredentials(
        service_account, 
        key_data=key_data
    )
    ee.Initialize(credentials)
    
except Exception as e:
    # Fallback for Local Run (Laptop)
    try:
        ee.Initialize()
    except Exception as e_inner:
        st.error(f"⚠️ Authentication Error: {e}")
        st.code(str(e)) # Show exact error to help debug
        st.info("Did you paste the Secrets correctly in the Streamlit Dashboard?")
        st.stop()

if 'calculated' not in st.session_state: st.session_state['calculated'] = False
if 'dates' not in st.session_state: st.session_state['dates'] = []
if 'roi' not in st.session_state: st.session_state['roi'] = None

# --- 4. FUNCTIONS (With Caching for Speed) ---
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

@st.cache_data(ttl=3600)
def compute_index(img, platform, index, formula=None):
    # Note: We cannot cache the EE object directly easily, but we cache the logic result
    if platform == "Sentinel-2 (Optical)":
        if index == '🛠️ Custom (Band Math)':
            map_b = {'B1':img.select('B1'), 'B2':img.select('B2'), 'B3':img.select('B3'), 'B4':img.select('B4'), 
                     'B8':img.select('B8'), 'B11':img.select('B11'), 'B12':img.select('B12')}
            return img.expression(formula, map_b).rename('Custom')
        map_i = {'NDVI': ['B8','B4'], 'GNDVI': ['B8','B3'], 'NDWI (Water)': ['B3','B8'], 'NDMI': ['B8','B11']}
        if index in map_i: return img.normalizedDifference(map_i[index]).rename(index.split()[0])
    elif platform == "Sentinel-1 (Radar)":
        if index == 'VV': return img.select('VV')
        if index == 'VH': return img.select('VH')
        if index == 'VH/VV Ratio': return img.select('VH').subtract(img.select('VV')).rename('Ratio')
    return img.select(0)

# Optimized to be faster (lower DPI)
@st.cache_data(ttl=3600)
def generate_static_map_img(vis_params, roi_bounds, dimensions=800):
    # Dummy function logic to fix caching - we can't cache EE objects well
    pass

def generate_static_map_display(image, roi, vis_params, title, cmap_colors):
    # Request smaller thumbnail for speed
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
    
    # Lower DPI for speed (80 instead of default)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
    ax.imshow(img_pil)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    img_w_px = img_pil.width
    scale_bar_px = img_w_px * 0.2
    scale_bar_m = width_m * 0.2
    scale_text = f"{scale_bar_m/1000:.1f} km" if scale_bar_m > 1000 else f"{int(scale_bar_m)} m"
    bar_x, bar_y = img_w_px * 0.05, img_pil.height * 0.95
    ax.add_patch(Rectangle((bar_x, bar_y - 5), scale_bar_px, 10, color='white'))
    ax.add_patch(Rectangle((bar_x, bar_y - 2), scale_bar_px, 4, color='black'))
    ax.text(bar_x + scale_bar_px/2, bar_y - 15, scale_text, ha='center', color='black', fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    cmap = mcolors.LinearSegmentedColormap.from_list("custom", cmap_colors)
    norm = mcolors.Normalize(vmin=vis_params['min'], vmax=vis_params['max'])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Index Value', fontsize=10)
    
    buf = BytesIO()
    plt.savefig(buf, format='jpg', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("Ni30 Analytics")
    st.markdown("---")
    
    with st.expander("📍 Region of Interest", expanded=True):
        roi_method = st.radio("Method", ["Upload KML", "Point & Buffer", "Manual Coordinates"])
        new_roi = None
        if roi_method == "Upload KML":
            kml = st.file_uploader("Upload KML", type=['kml'])
            if kml:
                kml.seek(0)
                new_roi = parse_kml(kml.read())
        elif roi_method == "Point & Buffer":
            lat = st.number_input("Latitude", 20.59)
            lon = st.number_input("Longitude", 78.96)
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
            # Reset if ROI changes
            if st.session_state['roi'] is None or new_roi.getInfo() != st.session_state['roi'].getInfo():
                st.session_state['roi'] = new_roi
                st.session_state['calculated'] = False
                st.session_state['dates'] = []
                st.toast("ROI Updated", icon="✅")

    with st.expander("🛰️ Configuration", expanded=True):
        platform = st.selectbox("Satellite", ["Sentinel-2 (Optical)", "Sentinel-1 (Radar)"])
        if platform == "Sentinel-2 (Optical)":
            idx = st.selectbox("Index", ['NDVI', 'GNDVI', 'NDWI (Water)', 'NDMI', '🛠️ Custom (Band Math)'])
            formula = st.text_input("Formula", "(B8-B4)/(B8+B4)") if 'Custom' in idx else ""
            vmin, vmax = (0.0, 0.8)
            pal_name = "Red-Yellow-Green"
            if 'Water' in idx: vmin, vmax, pal_name = -0.5, 0.5, "Blue-White-Green"
            
            c1, c2 = st.columns(2)
            vmin = c1.number_input("Min", vmin)
            vmax = c2.number_input("Max", vmax)
            pal_name = st.selectbox("Palette", ["Red-Yellow-Green", "Blue-White-Green", "Magma", "Viridis"])
            cloud = st.slider("Cloud %", 0, 30, 10)
            orbit = None
        else:
            idx = st.selectbox("Pol", ['VV', 'VH', 'VH/VV Ratio'])
            vmin, vmax = -25, -5
            pal_name = "Greyscale"
            orbit = st.radio("Orbit", ["DESCENDING", "ASCENDING", "BOTH"])
            cloud = 0
            formula = ""

    pal_map = {
        "Red-Yellow-Green": ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'],
        "Blue-White-Green": ['blue', 'white', 'green'],
        "Magma": ['black', 'purple', 'orange', 'white'],
        "Viridis": ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
        "Greyscale": ['black', 'white']
    }
    cur_palette = pal_map.get(pal_name, pal_map["Red-Yellow-Green"])

    with st.expander("📅 Time", expanded=True):
        mode = st.radio("Mode", ["Single", "Series"])
        if mode == "Single":
            d = st.date_input("Date", datetime.now()-timedelta(10))
            start, end = d.strftime("%Y-%m-%d"), (d+timedelta(1)).strftime("%Y-%m-%d")
        else:
            start = st.date_input("Start", datetime.now()-timedelta(60)).strftime("%Y-%m-%d")
            end = st.date_input("End", datetime.now()).strftime("%Y-%m-%d")

    st.markdown("###")
    if st.button("🚀 Calculate", type="primary"):
        if st.session_state['roi']:
            st.session_state.update({
                'calculated': True, 'platform': platform, 'idx': idx, 'start': start, 'end': end,
                'cloud': cloud, 'formula': formula, 'orbit': orbit,
                'vmin': vmin, 'vmax': vmax, 'palette': cur_palette
            })
            st.session_state['dates'] = []
        else:
            st.error("Select ROI first.")

# --- 6. MAIN CONTENT ---
st.markdown("""
<div class="navbar">
    <div class="navbar-title">Geospatial Ni30</div>
    <div style="color:#64748b; font-size:0.9rem;">Real-time Satellite Analysis</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state['calculated']:
    st.info("👈 Please configure the analysis in the sidebar.")
    m = geemap.Map(height=600, basemap="HYBRID")
    if st.session_state['roi']:
        m.centerObject(st.session_state['roi'], 12)
        m.addLayer(ee.Image().paint(st.session_state['roi'], 2, 3), {'palette': 'cyan'}, 'ROI')
    m.to_streamlit()
else:
    roi = st.session_state['roi']
    p = st.session_state
    
    with st.spinner("Processing..."):
        if p['platform'] == "Sentinel-2 (Optical)":
            col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterBounds(roi).filterDate(p['start'], p['end'])
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', p['cloud'])))
        else:
            col = (ee.ImageCollection('COPERNICUS/S1_GRD')
                   .filterBounds(roi).filterDate(p['start'], p['end'])
                   .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')))
            if p['orbit'] != "BOTH": col = col.filter(ee.Filter.eq('orbitProperties_pass', p['orbit']))

        # Add bands
        processed = col.map(lambda img: img.addBands(compute_index(img, p['platform'], p['idx'], p['formula'])))
        
        if not st.session_state['dates']:
            cnt = processed.size().getInfo()
            if cnt > 0:
                # Limit to first 20 dates to prevent slowdowns in Series mode
                dates_list = processed.aggregate_array('system:time_start').map(
                    lambda t: ee.Date(t).format('YYYY-MM-dd')).distinct().sort()
                st.session_state['dates'] = dates_list.slice(0, 50).getInfo() # Slice for safety
            else:
                st.warning("No images found.")
                st.session_state['calculated'] = False

    if st.session_state['dates']:
        dates = st.session_state['dates']
        
        # --- LAYOUT COLUMNS ---
        col_map, col_controls = st.columns([3, 1])
        
        # --- RIGHT COLUMN (CONTROLS) ---
        with col_controls:
            st.markdown('<div class="control-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📅 Select Date</div>', unsafe_allow_html=True)
            
            if mode == "Series":
                sel_date = st.selectbox("Available Images", dates, index=len(dates)-1, label_visibility="collapsed")
            else:
                sel_date = p['start']
                st.info(f"Single Date: {sel_date}")
                
            st.markdown(f"<div style='font-size:0.8rem; color:#64748b; margin-top:5px;'>{len(dates)} available</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Logic
            d_s = sel_date
            d_e = (datetime.strptime(sel_date, "%Y-%m-%d") + timedelta(1)).strftime("%Y-%m-%d")
            band = 'Custom' if 'Custom' in p['idx'] else p['idx'].split()[0]
            if 'Ratio' in p['idx']: band = 'Ratio'
            
            final_img = processed.filterDate(d_s, d_e).select(band).median().clip(roi)
            vis = {'min': p['vmin'], 'max': p['vmax'], 'palette': p['palette']}
            
            st.markdown('<div class="control-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">📥 Exports</div>', unsafe_allow_html=True)
            try:
                url = final_img.getDownloadURL({'scale': 10, 'region': roi, 'name': f"{band}_{sel_date}"})
                st.markdown(f"🔗 [Download GeoTIFF]({url})", unsafe_allow_html=True)
            except: st.caption("Area too large for link.")
            
            st.markdown("---")
            if st.button("☁️ Save to Drive", use_container_width=True):
                ee.batch.Export.image.toDrive(image=final_img, description=f"{band}_{sel_date}", 
                                              scale=10, region=roi, folder='GEE_Exports').start()
                st.toast("Export Started")
                
            st.markdown("---")
            if st.button("🎨 Generate Map (JPG)", use_container_width=True):
                with st.spinner("Generating..."):
                    # Using the new display function
                    buf = generate_static_map_display(final_img, roi, vis, f"{p['idx']} - {sel_date}", p['palette'])
                    st.download_button("⬇️ Download JPG", buf, f"Map_{sel_date}.jpg", "image/jpeg", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- LEFT COLUMN (MAP) ---
        with col_map:
            m = geemap.Map(height=700, basemap="HYBRID")
            m.centerObject(roi, 13)
            m.addLayer(final_img, vis, f"{p['idx']} ({sel_date})")
            m.add_colorbar(vis, label=p['idx'])
            m.to_streamlit()
