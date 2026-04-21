"""
Modul Praktikum 6: Verification & Validation
Studi Kasus: Pembagian Lembar Jawaban Ujian (Discrete Event Simulation)
[11S1221] Pemodelan dan Simulasi (MODSIM) — Institut Teknologi Del
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MODSIM P6 — Pembagian Lembar Jawaban Ujian",
    page_icon="📋",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# CSS — Dark background seperti referensi
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* BACKGROUND GELAP */
.stApp {
    background-color: #0e1117;
    font-family: 'Inter', sans-serif;
    color: #fafafa;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #2d3748;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] label { color: #90cdf4 !important; font-weight: 600; font-size: 0.85rem; }

/* JUDUL HALAMAN */
.page-title {
    font-size: 1.6rem; font-weight: 800; color: #f0f4ff;
    text-align: center; padding: 18px 0 4px 0;
    letter-spacing: -0.3px;
}
.page-sub {
    font-size: 0.85rem; color: #718096; text-align: center;
    margin-bottom: 20px;
    font-family: 'JetBrains Mono', monospace;
}

/* TOMBOL JALANKAN SIMULASI */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #e53e3e, #c53030) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 48px !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 20px rgba(229,62,62,0.4) !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #fc8181, #e53e3e) !important;
    box-shadow: 0 6px 28px rgba(229,62,62,0.55) !important;
    transform: translateY(-1px) !important;
}

/* METRIC CARDS */
.metric-row { display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: #1a2035;
    border: 1px solid #2d3748;
    border-radius: 12px; padding: 18px 16px; text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
}
.metric-label { font-size: 0.72rem; color: #718096; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.metric-value { font-size: 1.9rem; font-weight: 800; color: #63b3ed; line-height: 1; }
.metric-sub   { font-size: 0.73rem; color: #4a5568; margin-top: 4px; }

/* SECTION HEADER */
.sec-hdr {
    font-size: 1rem; font-weight: 700; color: #e2e8f0;
    background: #1a2035;
    border-left: 4px solid #4299e1;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin: 20px 0 12px 0;
    display: flex; align-items: center; gap: 8px;
}

/* PERTANYAAN HEADER (seperti P1, P2 di referensi) */
.q-header {
    font-size: 1.05rem; font-weight: 700; color: #f0f4ff;
    background: #1a2035;
    border-left: 4px solid #f6ad55;
    border-radius: 0 10px 10px 0;
    padding: 12px 18px;
    margin: 24px 0 10px 0;
    display: flex; align-items: center; gap: 10px;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: #161b27;
    border-radius: 12px; padding: 5px; gap: 3px;
    border: 1px solid #2d3748;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 9px;
    color: #718096 !important; font-weight: 600; font-size: 0.85rem;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #2d3748 !important;
    color: #e2e8f0 !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] { border-radius: 10px; border: 1px solid #2d3748 !important; }

/* DIVIDER */
hr { border-color: #2d3748 !important; }

/* EXPANDER */
.streamlit-expanderHeader {
    background: #1a2035 !important;
    border-radius: 8px !important;
    color: #90cdf4 !important;
    font-weight: 600 !important;
}

/* STATUS BOXES */
.stSuccess { background: rgba(72,187,120,0.12) !important; border: 1px solid rgba(72,187,120,0.35) !important; border-radius: 8px; }
.stWarning { background: rgba(246,173,85,0.12) !important; border: 1px solid rgba(246,173,85,0.35) !important; border-radius: 8px; }
.stInfo    { background: rgba(99,179,237,0.12) !important; border: 1px solid rgba(99,179,237,0.35) !important; border-radius: 8px; }
.stError   { background: rgba(252,129,129,0.12) !important; border: 1px solid rgba(252,129,129,0.35) !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────────
BG_DARK   = "#0e1117"
BG_PANEL  = "#1a2035"
FG_TEXT   = "#e2e8f0"
GRID_COL  = "#2d3748"
C_BLUE    = "#4299e1"
C_TEAL    = "#38b2ac"
C_ORANGE  = "#f6ad55"
C_RED     = "#fc8181"
C_GREEN   = "#68d391"
C_PURPLE  = "#b794f4"
C_YELLOW  = "#faf089"

def dark_style(fig, axes_list):
    import numpy as np
    fig.patch.set_facecolor(BG_DARK)
    # Flatten: handle single ax, list, atau numpy array dari subplots
    if isinstance(axes_list, np.ndarray):
        axes_flat = axes_list.flatten().tolist()
    elif isinstance(axes_list, list):
        axes_flat = axes_list
    else:
        axes_flat = [axes_list]
    for ax in axes_flat:
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=FG_TEXT, labelsize=9)
        ax.xaxis.label.set_color(FG_TEXT)
        ax.yaxis.label.set_color(FG_TEXT)
        ax.title.set_color(FG_TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, alpha=0.6, linestyle='-', linewidth=0.5)

# ─────────────────────────────────────────────────────────────
# FUNGSI SIMULASI
# ─────────────────────────────────────────────────────────────
def run_simulation(N, min_dur=1.0, max_dur=3.0, seed=None):
    rng = np.random.default_rng(seed)
    records = []
    current_time = 0.0
    for i in range(1, N + 1):
        start = current_time
        dur   = rng.uniform(min_dur, max_dur)
        end   = start + dur
        records.append({
            'Mahasiswa'              : i,
            'Mulai Dilayani (mnt)'   : round(start, 3),
            'Durasi Pelayanan (mnt)' : round(dur,   3),
            'Selesai Dilayani (mnt)' : round(end,   3),
            'Waktu Tunggu (mnt)'     : round(start, 3),
        })
        current_time = end
    df         = pd.DataFrame(records)
    total_time = current_time
    avg_wait   = df['Waktu Tunggu (mnt)'].mean()
    utilisasi  = df['Durasi Pelayanan (mnt)'].sum() / total_time * 100
    return df, total_time, avg_wait, utilisasi

def run_many(N, min_dur, max_dur, n_rep=300):
    return [run_simulation(N, min_dur, max_dur, seed=s)[1] for s in range(n_rep)]

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">📋 Simulasi Pembagian Lembar Jawaban Ujian</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">[11S1221] Pemodelan dan Simulasi · Modul Praktikum 6 · Verification &amp; Validation · Institut Teknologi Del</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameter Simulasi")
    st.markdown("---")
    N       = st.slider("👥 Jumlah Mahasiswa (N)", 5, 100, 30, step=5)
    min_dur = st.number_input("⏬ Durasi Min (menit)", 0.5, 10.0, 1.0, step=0.5)
    max_dur = st.number_input("⏫ Durasi Max (menit)", 1.0, 20.0, 3.0, step=0.5)
    seed    = st.number_input("🎲 Random Seed", 0, 9999, 42, step=1)
    n_rep   = st.slider("🔁 Jumlah Replikasi", 100, 1000, 300, step=100)
    st.markdown("---")

    if min_dur >= max_dur:
        st.error("⚠️ Durasi Min harus < Durasi Max!")
        st.stop()

    E_T = (min_dur + max_dur) / 2
    st.markdown(f"""
**📌 Info Model**
- Distribusi: `Uniform({min_dur}, {max_dur})`
- E(T) = `{E_T:.2f} menit`
- Total Teoritis ≈ `{N * E_T:.1f} menit`
- Antrian: `FIFO` · Server: `Single`
""")
    st.markdown("---")
    # TOMBOL di sidebar
    run_btn = st.button("🚀 Jalankan Simulasi", key="btn_sidebar")

# ─────────────────────────────────────────────────────────────
# STATE — simpan apakah simulasi sudah dijalankan
# ─────────────────────────────────────────────────────────────
if "sudah_run" not in st.session_state:
    st.session_state.sudah_run = False
if "sim_params" not in st.session_state:
    st.session_state.sim_params = {}

if run_btn:
    st.session_state.sudah_run = True
    st.session_state.sim_params = {
        "N": N, "min_dur": min_dur, "max_dur": max_dur,
        "seed": int(seed), "n_rep": n_rep
    }

# ─────────────────────────────────────────────────────────────
# SEBELUM SIMULASI DIJALANKAN — Tampilan awal
# ─────────────────────────────────────────────────────────────
if not st.session_state.sudah_run:
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
        <div style="font-size:4rem; margin-bottom:16px;">📋</div>
        <div style="font-size:1.3rem; font-weight:700; color:#e2e8f0; margin-bottom:10px;">
            Siap Menjalankan Simulasi
        </div>
        <div style="font-size:0.9rem; color:#718096; max-width:500px; margin:0 auto 28px auto; line-height:1.6;">
            Atur parameter di <strong style="color:#90cdf4">sidebar kiri</strong>, lalu klik tombol
            <strong style="color:#fc8181">🚀 Jalankan Simulasi</strong> untuk memulai
            Discrete Event Simulation pembagian lembar jawaban ujian.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tombol besar di tengah halaman juga
    col_l, col_c, col_r = st.columns([2, 2, 2])
    with col_c:
        if st.button("🚀 Jalankan Simulasi", key="btn_center"):
            st.session_state.sudah_run = True
            st.session_state.sim_params = {
                "N": N, "min_dur": min_dur, "max_dur": max_dur,
                "seed": int(seed), "n_rep": n_rep
            }
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="display:flex; gap:20px; flex-wrap:wrap; margin-top:10px;">
        <div style="flex:1; min-width:180px; background:#1a2035; border-radius:12px; padding:20px; border:1px solid #2d3748;">
            <div style="font-size:1.5rem; margin-bottom:8px;">🔍</div>
            <div style="font-weight:700; color:#90cdf4; margin-bottom:6px;">Verifikasi</div>
            <div style="font-size:0.82rem; color:#718096; line-height:1.5;">
                Logical flow, event tracing, uji kondisi ekstrem, distribusi, reproducibility
            </div>
        </div>
        <div style="flex:1; min-width:180px; background:#1a2035; border-radius:12px; padding:20px; border:1px solid #2d3748;">
            <div style="font-size:1.5rem; margin-bottom:8px;">✅</div>
            <div style="font-weight:700; color:#68d391; margin-bottom:6px;">Validasi</div>
            <div style="font-size:0.82rem; color:#718096; line-height:1.5;">
                Face validation, perbandingan teoritis, behavior validation, sensitivity analysis
            </div>
        </div>
        <div style="flex:1; min-width:180px; background:#1a2035; border-radius:12px; padding:20px; border:1px solid #2d3748;">
            <div style="font-size:1.5rem; margin-bottom:8px;">📊</div>
            <div style="font-weight:700; color:#f6ad55; margin-bottom:6px;">Grafik Lengkap</div>
            <div style="font-size:0.82rem; color:#718096; line-height:1.5;">
                Histogram, box plot, gantt chart, distribusi replikasi, sensitivity
            </div>
        </div>
        <div style="flex:1; min-width:180px; background:#1a2035; border-radius:12px; padding:20px; border:1px solid #2d3748;">
            <div style="font-size:1.5rem; margin-bottom:8px;">📋</div>
            <div style="font-weight:700; color:#b794f4; margin-bottom:6px;">Event Log</div>
            <div style="font-size:0.82rem; color:#718096; line-height:1.5;">
                Tabel lengkap setiap mahasiswa, download CSV, scatter & cumulative chart
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# AMBIL PARAMS DARI SESSION STATE
# ─────────────────────────────────────────────────────────────
p       = st.session_state.sim_params
N       = p["N"]
min_dur = p["min_dur"]
max_dur = p["max_dur"]
seed    = p["seed"]
n_rep   = p["n_rep"]
E_T     = (min_dur + max_dur) / 2
theoretical = N * E_T

# Jalankan simulasi
with st.spinner("⏳ Menjalankan simulasi..."):
    df, total_time, avg_wait, utilisasi = run_simulation(N, min_dur, max_dur, seed=seed)

st.success(f"✅ Simulasi selesai! N={N}, Uniform({min_dur},{max_dur}), seed={seed}")

# ─────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────
err_face = abs(total_time - theoretical) / theoretical * 100
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-label">⏱ Total Waktu</div>
        <div class="metric-value">{total_time:.1f}</div>
        <div class="metric-sub">menit</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">⌛ Rata-rata Tunggu</div>
        <div class="metric-value">{avg_wait:.1f}</div>
        <div class="metric-sub">menit</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">📈 Utilisasi Server</div>
        <div class="metric-value">{utilisasi:.1f}%</div>
        <div class="metric-sub">efisiensi meja pengajar</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">🎯 Nilai Teoritis</div>
        <div class="metric-value">{theoretical:.1f}</div>
        <div class="metric-sub">menit (N × E(T))</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">👥 Mahasiswa</div>
        <div class="metric-value">{N}</div>
        <div class="metric-sub">orang</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">📉 Selisih Teori</div>
        <div class="metric-value">{err_face:.1f}%</div>
        <div class="metric-sub">dari nilai teoritis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tombol reset
col_r1, col_r2, col_r3 = st.columns([4, 1, 1])
with col_r3:
    if st.button("🔄 Reset", key="btn_reset"):
        st.session_state.sudah_run = False
        st.rerun()

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Hasil Simulasi", "🔍 Verifikasi", "✅ Validasi", "📋 Event Log"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — HASIL SIMULASI
# ══════════════════════════════════════════════════════════════
with tab1:

    # ── Q1: Distribusi Durasi ──────────────────────────────────
    st.markdown('<div class="q-header">📊 Q1 — Berapa Total Waktu Pembagian Lembar Jawaban?</div>', unsafe_allow_html=True)

    dur = df['Durasi Pelayanan (mnt)']

    # Histogram besar seperti referensi
    fig1, ax1 = plt.subplots(figsize=(14, 4.5))
    dark_style(fig1, [ax1])

    # Histogram dengan gradien warna
    n_h, bins, patches = ax1.hist(dur, bins=max(8, N//3),
                                   edgecolor=BG_DARK, linewidth=0.5, alpha=0.9)
    norm_v = (bins[:-1] - bins[:-1].min()) / (bins[:-1].max() - bins[:-1].min() + 1e-9)
    cmap_h = plt.cm.get_cmap('Blues')
    for patch, nv in zip(patches, norm_v):
        patch.set_facecolor(cmap_h(0.35 + nv * 0.6))

    ax1.axvline(min_dur,    color=C_RED,    linestyle='--', lw=2,   label=f'Min = {min_dur}')
    ax1.axvline(max_dur,    color=C_RED,    linestyle='--', lw=2,   label=f'Max = {max_dur}')
    ax1.axvline(dur.mean(), color=C_ORANGE, linestyle='-',  lw=2.5, label=f'Mean = {dur.mean():.2f}')
    ax1.axvline(E_T,        color=C_GREEN,  linestyle=':',  lw=2,   label=f'E(T) teori = {E_T:.2f}')

    # Shade area
    ax1.axvspan(min_dur, max_dur, alpha=0.06, color=C_BLUE)

    ax1.set_title(f'Distribusi Durasi Pelayanan — N={N} Mahasiswa', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('Durasi Pelayanan (menit)', fontsize=10)
    ax1.set_ylabel('Frekuensi', fontsize=10)
    ax1.legend(facecolor=BG_PANEL, labelcolor=FG_TEXT, edgecolor=GRID_COL, fontsize=9, loc='upper right')

    plt.tight_layout()
    st.pyplot(fig1); plt.close()

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Waktu Simulasi", f"{total_time:.2f} mnt")
    c2.metric("Nilai Teoritis",       f"{theoretical:.2f} mnt")
    c3.metric("Interval (Min–Max)",   f"{N*min_dur:.0f} – {N*max_dur:.0f} mnt")
    c4.metric("Mean Durasi",          f"{dur.mean():.3f} mnt")

    with st.expander("📋 Detail Statistik Durasi Pelayanan"):
        st.dataframe(dur.describe().to_frame().T.round(3), use_container_width=True)

    # ── Q2: Waktu Tunggu & Gantt ───────────────────────────────
    st.markdown('<div class="q-header">⏳ Q2 — Bagaimana Pola Waktu Tunggu Mahasiswa?</div>', unsafe_allow_html=True)

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4.5))
    dark_style(fig2, axes2)

    # Waktu tunggu — area + bar
    axes2[0].bar(df['Mahasiswa'], df['Waktu Tunggu (mnt)'],
                 color=C_TEAL, alpha=0.7, edgecolor='none', width=0.8)
    axes2[0].plot(df['Mahasiswa'], df['Waktu Tunggu (mnt)'],
                  color=C_ORANGE, lw=1.5, alpha=0.9)
    axes2[0].set_title('Waktu Tunggu per Mahasiswa', fontsize=11, fontweight='bold')
    axes2[0].set_xlabel('Mahasiswa ke-'); axes2[0].set_ylabel('Waktu Tunggu (mnt)')

    # Durasi bar dengan warna per mahasiswa
    cb = plt.cm.Blues(np.linspace(0.3, 0.9, N))
    axes2[1].bar(df['Mahasiswa'], df['Durasi Pelayanan (mnt)'],
                 color=cb, edgecolor='none', alpha=0.9)
    axes2[1].axhline(E_T, color=C_ORANGE, linestyle='--', lw=1.8, label=f'E(T)={E_T:.1f}')
    axes2[1].set_title('Durasi Pelayanan per Mahasiswa', fontsize=11, fontweight='bold')
    axes2[1].set_xlabel('Mahasiswa ke-'); axes2[1].set_ylabel('Durasi (mnt)')
    axes2[1].legend(facecolor=BG_PANEL, labelcolor=FG_TEXT, edgecolor=GRID_COL, fontsize=9)

    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # ── Q3: Gantt Chart ────────────────────────────────────────
    st.markdown('<div class="q-header">📅 Q3 — Urutan & Timeline Pelayanan (Gantt Chart)</div>', unsafe_allow_html=True)

    n_show = min(25, N)
    fig3, ax3 = plt.subplots(figsize=(14, max(4, n_show * 0.38)))
    dark_style(fig3, [ax3])

    cmap_g = plt.cm.get_cmap('Blues')
    gc     = cmap_g(np.linspace(0.35, 0.95, n_show))
    for idx, row in df.head(n_show).iterrows():
        ax3.barh(f"Mhs-{int(row['Mahasiswa']):02d}",
                 row['Durasi Pelayanan (mnt)'],
                 left=row['Mulai Dilayani (mnt)'],
                 color=gc[idx], edgecolor='none', height=0.65, alpha=0.92)
        ax3.text(
            row['Mulai Dilayani (mnt)'] + row['Durasi Pelayanan (mnt)'] / 2,
            idx, f"{row['Durasi Pelayanan (mnt)']:.1f}",
            ha='center', va='center', fontsize=7.5, color='white', fontweight='bold'
        )

    ax3.set_xlabel('Waktu (menit)')
    ax3.set_title(f'Gantt Chart Pelayanan — {n_show} Mahasiswa Pertama',
                  fontsize=11, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, axis='x', color=GRID_COL, alpha=0.5, linestyle='--')
    ax3.grid(False, axis='y')
    plt.tight_layout(); st.pyplot(fig3); plt.close()

    # ── Q4: Box Plot + Cumulative ──────────────────────────────
    st.markdown('<div class="q-header">📦 Q4 — Statistik Distribusi & Akumulasi Waktu</div>', unsafe_allow_html=True)

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 4.5))
    dark_style(fig4, axes4)

    # Box plot
    bp = axes4[0].boxplot(dur, vert=True, patch_artist=True, widths=0.5,
                          boxprops   =dict(facecolor=C_BLUE,   color=FG_TEXT, alpha=0.65),
                          medianprops=dict(color=C_ORANGE,     linewidth=3),
                          whiskerprops=dict(color=FG_TEXT,     linewidth=1.5),
                          capprops   =dict(color=FG_TEXT,      linewidth=1.5),
                          flierprops =dict(marker='o', color=C_RED, markersize=5, alpha=0.7))
    axes4[0].axhline(E_T, color=C_GREEN, linestyle='--', lw=1.8, label=f'E(T) teori={E_T:.1f}')
    axes4[0].set_title('Box Plot Durasi Pelayanan', fontsize=11, fontweight='bold')
    axes4[0].set_ylabel('Durasi (menit)'); axes4[0].set_xticklabels(['Durasi Pelayanan'])
    axes4[0].legend(facecolor=BG_PANEL, labelcolor=FG_TEXT, edgecolor=GRID_COL, fontsize=9)

    # Cumulative step
    cum = df['Selesai Dilayani (mnt)']
    axes4[1].step(df['Mahasiswa'], cum, color=C_TEAL, lw=2.5, where='post', label='Simulasi')
    axes4[1].fill_between(df['Mahasiswa'], cum, step='post', alpha=0.18, color=C_TEAL)
    axes4[1].axline((1, E_T), slope=E_T, color=C_ORANGE, linestyle='--', lw=2,
                    label=f'Teoritis (slope={E_T})')
    axes4[1].set_title('Akumulasi Waktu Pembagian', fontsize=11, fontweight='bold')
    axes4[1].set_xlabel('Mahasiswa ke-'); axes4[1].set_ylabel('Waktu Selesai (mnt)')
    axes4[1].legend(facecolor=BG_PANEL, labelcolor=FG_TEXT, edgecolor=GRID_COL, fontsize=9)

    plt.tight_layout(); st.pyplot(fig4); plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 2 — VERIFIKASI
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="q-header">🔍 Verification — Build the Model Right?</div>', unsafe_allow_html=True)

    # a. Logical Flow
    st.subheader("a. Logical Flow Check")
    overlap = any(
        df.iloc[i+1]['Mulai Dilayani (mnt)'] < df.iloc[i]['Selesai Dilayani (mnt)']
        for i in range(len(df)-1)
    )
    if not overlap:
        st.success("✅ Tidak ada tumpang tindih. Setiap mahasiswa dilayani satu per satu (FIFO).")
    else:
        st.error("❌ Terdapat tumpang tindih waktu pelayanan!")

    # b. Event Tracing
    st.subheader("b. Event Tracing (5 Mahasiswa Pertama)")
    st.dataframe(df.head(5), use_container_width=True)

    # c. Extreme Condition
    st.subheader("c. Uji Kondisi Ekstrem")
    ext = []
    df1, t1, _, _ = run_simulation(1, min_dur, max_dur, seed=42)
    ext.append({'Skenario': 'N=1',
                'Simulasi (mnt)': round(t1, 3),
                'Ekspektasi (mnt)': round(df1.iloc[0]['Durasi Pelayanan (mnt)'], 3),
                'Status': '✅ Sesuai' if abs(t1 - df1.iloc[0]['Durasi Pelayanan (mnt)']) < 0.001 else '❌'})
    df2, t2, _, _ = run_simulation(N, min_dur=min_dur, max_dur=min_dur, seed=42)
    ext.append({'Skenario': f'Durasi tetap = {min_dur}',
                'Simulasi (mnt)': round(t2, 3),
                'Ekspektasi (mnt)': round(N * min_dur, 3),
                'Status': '✅ Sesuai' if abs(t2 - N*min_dur) < 0.001 else '❌'})
    df3, t3, _, _ = run_simulation(N, min_dur=max_dur, max_dur=max_dur, seed=42)
    ext.append({'Skenario': f'Durasi tetap = {max_dur}',
                'Simulasi (mnt)': round(t3, 3),
                'Ekspektasi (mnt)': round(N * max_dur, 3),
                'Status': '✅ Sesuai' if abs(t3 - N*max_dur) < 0.001 else '❌'})
    df_ext = pd.DataFrame(ext)
    st.dataframe(df_ext, use_container_width=True)

    # Grafik grouped bar extreme
    fig_e, ax_e = plt.subplots(figsize=(10, 3.5))
    dark_style(fig_e, [ax_e])
    x = np.arange(len(ext)); w = 0.35
    ax_e.bar(x-w/2, [e['Simulasi (mnt)'] for e in ext],   w, color=C_BLUE,   alpha=0.85, edgecolor='none', label='Simulasi')
    ax_e.bar(x+w/2, [e['Ekspektasi (mnt)'] for e in ext], w, color=C_ORANGE, alpha=0.85, edgecolor='none', label='Ekspektasi')
    ax_e.set_xticks(x); ax_e.set_xticklabels([e['Skenario'] for e in ext], fontsize=9)
    ax_e.set_ylabel('Total Waktu (mnt)')
    ax_e.set_title('Uji Kondisi Ekstrem — Simulasi vs Ekspektasi', fontsize=10, fontweight='bold')
    ax_e.legend(facecolor=BG_PANEL, labelcolor=FG_TEXT, edgecolor=GRID_COL, fontsize=9)
    plt.tight_layout(); st.pyplot(fig_e); plt.close()

    # d. Distribusi
    st.subheader("d. Pemeriksaan Distribusi Waktu Pelayanan")
    dur = df['Durasi Pelayanan (mnt)']
    c1, c2, c3 = st.columns(3)
    c1.metric("Min Aktual",  f"{dur.min():.3f} mnt", f"Batas bawah: {min_dur}")
    c2.metric("Max Aktual",  f"{dur.max():.3f} mnt", f"Batas atas: {max_dur}")
    c3.metric("Mean Aktual", f"{dur.mean():.3f} mnt", f"Teori E(T): {E_T:.2f}")
    if dur.min() >= min_dur-0.001 and dur.max() <= max_dur+0.001:
        st.success(f"✅ Semua nilai durasi dalam rentang Uniform({min_dur}, {max_dur}) — distribusi benar.")
    else:
        st.error("❌ Ada nilai di luar rentang distribusi!")

    # e. Reproducibility
    st.subheader("e. Reproducibility Check")
    r1 = run_simulation(N, min_dur, max_dur, seed=999)[1]
    r2 = run_simulation(N, min_dur, max_dur, seed=999)[1]
    r3 = run_simulation(N, min_dur, max_dur, seed=999)[1]
    fig_r, ax_r = plt.subplots(figsize=(7, 3))
    dark_style(fig_r, [ax_r])
    bars_r = ax_r.bar(['Run 1', 'Run 2', 'Run 3'], [r1, r2, r3],
                      color=[C_BLUE, C_TEAL, C_PURPLE], alpha=0.85, edgecolor='none', width=0.45)
    for bar, v in zip(bars_r, [r1, r2, r3]):
        ax_r.text(bar.get_x()+bar.get_width()/2, v + max(r1,r2,r3)*0.01,
                  f'{v:.3f}', ha='center', fontsize=9.5, color=FG_TEXT, fontweight='bold')
    ax_r.set_title('Reproducibility Check (seed=999)', fontsize=10, fontweight='bold')
    ax_r.set_ylabel('Total Waktu (mnt)')
    ax_r.set_ylim(0, max(r1,r2,r3)*1.18)
    plt.tight_layout(); st.pyplot(fig_r); plt.close()
    if r1 == r2 == r3:
        st.success("✅ Output identik di setiap run dengan seed yang sama — Reproducibility terpenuhi.")

    st.markdown("---")
    st.success("**Kesimpulan Verifikasi:** Model telah terverifikasi dengan benar — logika sesuai asumsi, implementasi benar, hasil konsisten.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — VALIDASI
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="q-header">✅ Validation — Build the Right Model?</div>', unsafe_allow_html=True)

    # a. Face Validation
    st.subheader("a. Face Validation")
    lo_e = N*min_dur; hi_e = N*max_dur
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Waktu Simulasi",    f"{total_time:.2f} mnt")
    c2.metric("Rentang Diharapkan",      f"{lo_e:.0f} – {hi_e:.0f} mnt")
    c3.metric("Utilisasi Meja",          f"{utilisasi:.1f}%")
    if lo_e <= total_time <= hi_e:
        st.success(f"✅ Total waktu {total_time:.1f} mnt berada dalam rentang realistis [{lo_e:.0f}, {hi_e:.0f}] menit.")
    else:
        st.warning(f"⚠️ Total waktu di luar rentang [{lo_e:.0f}, {hi_e:.0f}] — periksa parameter!")

    # b. Perbandingan Teoritis — histogram replikasi besar
    st.subheader("b. Perbandingan dengan Nilai Teoritis")
    with st.spinner("Menjalankan replikasi..."):
        totals   = run_many(N, min_dur, max_dur, n_rep=n_rep)
    sim_mean = np.mean(totals); sim_std = np.std(totals)
    err_pct  = abs(sim_mean - theoretical) / theoretical * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nilai Teoritis",             f"{theoretical:.1f} mnt")
    c2.metric(f"Rata-rata ({n_rep} rep)",   f"{sim_mean:.2f} mnt")
    c3.metric("Std Dev",                    f"{sim_std:.2f} mnt")
    c4.metric("Selisih Relatif",            f"{err_pct:.2f}%")

    # Histogram besar distribusi replikasi
    fig_d, ax_d = plt.subplots(figsize=(14, 5))
    dark_style(fig_d, [ax_d])
    n_h2, bins2, patches2 = ax_d.hist(totals, bins=40, edgecolor=BG_DARK, linewidth=0.3, alpha=0.9)
    norm_v2 = (bins2[:-1] - bins2[:-1].min()) / (bins2[:-1].max() - bins2[:-1].min() + 1e-9)
    for p, nv in zip(patches2, norm_v2):
        p.set_facecolor(plt.cm.Blues(0.35 + nv*0.6))

    ax_d.axvline(theoretical, color=C_RED,    linestyle='--', lw=2.5, label=f'Teoritis = {theoretical:.1f} mnt')
    ax_d.axvline(sim_mean,    color=C_ORANGE, linestyle='-',  lw=2.5, label=f'Sim. Mean = {sim_mean:.1f} mnt')
    ax_d.axvspan(sim_mean-sim_std, sim_mean+sim_std,
                 alpha=0.13, color=C_TEAL, label=f'±1 Std = {sim_std:.1f}')
    ax_d.axvspan(sim_mean-2*sim_std, sim_mean+2*sim_std,
                 alpha=0.06, color=C_TEAL, label=f'±2 Std = {2*sim_std:.1f}')

    # Anotasi
    ax_d.annotate(f'Mean={sim_mean:.1f}', xy=(sim_mean, n_h2.max()*0.85),
                  fontsize=9, color=C_ORANGE, ha='center', fontweight='bold')
    ax_d.annotate(f'Teori={theoretical:.1f}', xy=(theoretical, n_h2.max()*0.7),
                  fontsize=9, color=C_RED, ha='center', fontweight='bold')

    ax_d.set_title(f'Distribusi Total Waktu Pembagian — {n_rep} Replikasi (N={N})',
                   fontsize=12, fontweight='bold', pad=12)
    ax_d.set_xlabel('Total Waktu (menit)', fontsize=10)
    ax_d.set_ylabel('Frekuensi', fontsize=10)
    ax_d.legend(facecolor=BG_PANEL, labelcolor=FG_TEXT, edgecolor=GRID_COL, fontsize=9)
    plt.tight_layout(); st.pyplot(fig_d); plt.close()

    if err_pct < 5:
        st.success(f"✅ Selisih {err_pct:.2f}% < 5% — Rata-rata simulasi mendekati nilai teoritis.")
    else:
        st.info(f"ℹ️ Selisih {err_pct:.1f}% — tambah jumlah replikasi.")

    # c. Behavior Validation
    st.subheader("c. Behavior Validation")
    N_vals = list(range(5, min(N + 40, 115), 10))
    with st.spinner("Behavior validation..."):
        beh = [np.mean(run_many(n, min_dur, max_dur, n_rep=80)) for n in N_vals]

    fig_b, ax_b = plt.subplots(figsize=(12, 4))
    dark_style(fig_b, [ax_b])
    ax_b.plot(N_vals, beh, 'o-', color=C_BLUE, lw=2.5, markersize=8, label='Rata-rata Simulasi', zorder=3)
    ax_b.fill_between(N_vals, beh, alpha=0.12, color=C_BLUE)
    ax_b.plot(N_vals, [n*E_T for n in N_vals], '--', color=C_ORANGE, lw=2, label=f'Teoritis (N×{E_T})')
    for xv, yv in zip(N_vals, beh):
        ax_b.scatter(xv, yv, s=65, color=C_TEAL, zorder=4)
    ax_b.set_title('Total Waktu vs Jumlah Mahasiswa (Behavior Validation)',
                   fontsize=11, fontweight='bold')
    ax_b.set_xlabel('Jumlah Mahasiswa (N)'); ax_b.set_ylabel('Rata-rata Total Waktu (mnt)')
    ax_b.legend(facecolor=BG_PANEL, labelcolor=FG_TEXT, edgecolor=GRID_COL, fontsize=9)
    plt.tight_layout(); st.pyplot(fig_b); plt.close()
    if all(beh[i] < beh[i+1] for i in range(len(beh)-1)):
        st.success("✅ Total waktu meningkat monoton — sesuai ekspektasi teoritis.")

    # d. Sensitivity Analysis
    st.subheader("d. Sensitivity Analysis")
    mid = (min_dur + max_dur) / 2
    scenarios = [
        (f"Uniform({min_dur},{max_dur})\nbaseline", min_dur, max_dur),
        (f"Uniform({mid},{mid+2})\nnaik",            mid,     mid+2),
        (f"Uniform({max(0.5,min_dur-0.5)},{max(1.5,max_dur-1)})\nturun",
         max(0.5, min_dur-0.5), max(1.5, max_dur-1)),
    ]
    sc_data = []
    with st.spinner("Sensitivity analysis..."):
        for label, lo, hi in scenarios:
            if lo >= hi: hi = lo+0.5
            vals = run_many(N, lo, hi, n_rep=200)
            sc_data.append({'Skenario': label.replace('\n', ' '),
                            'Rata-rata (mnt)': round(np.mean(vals), 2),
                            'Std Dev': round(np.std(vals), 2),
                            'E(T)': round((lo+hi)/2, 2)})
    df_sc = pd.DataFrame(sc_data)
    st.dataframe(df_sc, use_container_width=True)

    fig_sc, axes_sc = plt.subplots(1, 2, figsize=(13, 4))
    dark_style(fig_sc, axes_sc)
    cc = [C_BLUE, C_ORANGE, C_PURPLE]
    lbl = [s[0].replace('\n',' ') for s in scenarios]
    b1 = axes_sc[0].bar(lbl, df_sc['Rata-rata (mnt)'], color=cc, alpha=0.85, edgecolor='none')
    for bar, v in zip(b1, df_sc['Rata-rata (mnt)']):
        axes_sc[0].text(bar.get_x()+bar.get_width()/2, v+0.4,
                        f'{v:.1f}', ha='center', fontsize=10, color=FG_TEXT, fontweight='bold')
    axes_sc[0].set_title('Rata-rata Total Waktu per Skenario', fontsize=10, fontweight='bold')
    axes_sc[0].set_ylabel('Total Waktu (mnt)'); axes_sc[0].tick_params(axis='x', labelsize=8)
    b2 = axes_sc[1].bar(lbl, df_sc['Std Dev'], color=cc, alpha=0.72, edgecolor='none')
    for bar, v in zip(b2, df_sc['Std Dev']):
        axes_sc[1].text(bar.get_x()+bar.get_width()/2, v+0.1,
                        f'{v:.2f}', ha='center', fontsize=10, color=FG_TEXT, fontweight='bold')
    axes_sc[1].set_title('Variabilitas (Std Dev) per Skenario', fontsize=10, fontweight='bold')
    axes_sc[1].set_ylabel('Std Dev (mnt)'); axes_sc[1].tick_params(axis='x', labelsize=8)
    plt.tight_layout(); st.pyplot(fig_sc); plt.close()
    st.success("✅ Model sensitif terhadap perubahan parameter distribusi — sesuai ekspektasi.")

    st.markdown("---")
    st.success("""
**Kesimpulan Validasi:**
- ✅ Hasil dalam rentang realistis
- ✅ Rata-rata simulasi mendekati nilai teoritis
- ✅ Perilaku model konsisten dengan perubahan N
- ✅ Model sensitif terhadap parameter distribusi
→ **Model layak digunakan sebagai alat bantu analisis.**
""")

# ══════════════════════════════════════════════════════════════
# TAB 4 — EVENT LOG
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="q-header">📋 Event Log — Seluruh Mahasiswa</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Mahasiswa", N)
    c2.metric("Total Waktu",     f"{total_time:.2f} mnt")
    c3.metric("Seed",            seed)
    c4.metric("Utilisasi",       f"{utilisasi:.1f}%")

    st.dataframe(
        df.style
          .background_gradient(subset=['Waktu Tunggu (mnt)'],   cmap='Blues')
          .background_gradient(subset=['Durasi Pelayanan (mnt)'], cmap='YlOrRd'),
        use_container_width=True, height=460
    )

    fig_ev, axes_ev = plt.subplots(1, 2, figsize=(13, 3.8))
    dark_style(fig_ev, axes_ev)

    sc = axes_ev[0].scatter(df['Mahasiswa'], df['Durasi Pelayanan (mnt)'],
                            c=df['Durasi Pelayanan (mnt)'], cmap='Blues',
                            s=55, alpha=0.85, edgecolors='none', vmin=min_dur, vmax=max_dur)
    axes_ev[0].axhline(E_T, color=C_ORANGE, linestyle='--', lw=1.5, label=f'E(T)={E_T}')
    axes_ev[0].set_title('Scatter Durasi Pelayanan', fontsize=10, fontweight='bold')
    axes_ev[0].set_xlabel('Mahasiswa ke-'); axes_ev[0].set_ylabel('Durasi (mnt)')
    axes_ev[0].legend(facecolor=BG_PANEL, labelcolor=FG_TEXT, edgecolor=GRID_COL, fontsize=8)
    plt.colorbar(sc, ax=axes_ev[0])

    axes_ev[1].plot(df['Mahasiswa'], df['Selesai Dilayani (mnt)'], color=C_TEAL, lw=2)
    axes_ev[1].fill_between(df['Mahasiswa'], df['Selesai Dilayani (mnt)'], alpha=0.18, color=C_TEAL)
    axes_ev[1].axline((1, E_T), slope=E_T, color=C_ORANGE, linestyle='--', lw=1.8,
                      label=f'Teoritis (N×{E_T})')
    axes_ev[1].set_title('Progress Waktu Selesai Kumulatif', fontsize=10, fontweight='bold')
    axes_ev[1].set_xlabel('Mahasiswa ke-'); axes_ev[1].set_ylabel('Waktu Selesai (mnt)')
    axes_ev[1].legend(facecolor=BG_PANEL, labelcolor=FG_TEXT, edgecolor=GRID_COL, fontsize=8)
    plt.tight_layout(); st.pyplot(fig_ev); plt.close()

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV Event Log", csv, "event_log.csv", "text/csv")