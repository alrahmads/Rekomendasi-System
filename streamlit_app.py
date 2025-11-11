import streamlit as st
import pandas as pd
import json
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# === CONFIGURASI DASHBOARD ===
# ============================
st.set_page_config(layout="wide")
st.title("Rekomendasi Lowongan Pekerjaan Pengolahan Data")

# ============================
# === LOAD & PREPARE DATA ===
# ============================
df = pd.read_csv('Data/Data_Clean.csv')

# Bersihkan data
df.dropna(subset=["Job Title", "Job Qualifications", "Type", "Work Hours", "Kota", "URL"], inplace=True)
df["Job Qualifications"] = df["Job Qualifications"].fillna("")

# Gabungkan fitur untuk TF-IDF
df["combined_features"] = (
    df["Job Qualifications"] + " " + df["Job Title"] + " " + df["Type"] + " " + df["Work Hours"]
)

# Load koordinat kota dari JSON
with open("Data/location.json") as f:
    location_data = json.load(f)

# ============================
# === FUNGSI BANTUAN ===
# ============================

def parse_point(geo_str):
    """Parsing string 'POINT (lng lat)' menjadi tuple (lat, lng)."""
    coord_str = geo_str.replace("POINT (", "").replace(")", "")
    lng_str, lat_str = coord_str.split()
    return float(lat_str), float(lng_str)

def generate_map_with_counts(results):
    """Buat peta Folium dengan jumlah lowongan per kota."""
    counts = results["Kota"].value_counts()
    m = folium.Map(location=[-0.789275, 113.921327], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m)

    for kota, count in counts.items():
        kota_data = next((loc for loc in location_data if loc["Kota"].lower() == kota.lower()), None)
        if kota_data:
            lat, lng = parse_point(kota_data["Georeferenced"])
            popup_text = f"{kota}: {count} lowongan"
            folium.Marker(
                location=[lat, lng],
                popup=popup_text,
                tooltip=popup_text,
                icon=folium.Icon(color="blue", icon="briefcase")
            ).add_to(marker_cluster)
    return m

# ============================
# === FORM INPUT STREAMLIT ===
# ============================

st.subheader("Masukkan Preferensi & Kualifikasi Anda")

# Pilihan posisi pekerjaan
job_options = sorted(df["Job Title"].dropna().unique())
job_options.insert(0, "Choose an option")
job_preference = st.selectbox("Posisi yang dicari", options=job_options)

# Pilihan tipe pekerjaan
job_types = sorted(df["Type"].dropna().unique())
job_types.insert(0, "Choose an option")
selected_type = st.selectbox("Tipe Pekerjaan", options=job_types)

# Pilihan jam kerja
work_hours = sorted(df["Work Hours"].dropna().unique())
work_hours.insert(0, "Choose an option")
selected_hours = st.selectbox("Jam Kerja", options=work_hours)

# Pilihan kualifikasi
all_qualifications = sorted(df["Job Qualifications"].str.split(",").explode().str.strip().dropna().unique())
selected_qualifications = st.multiselect("Kualifikasi Anda", all_qualifications)

# ============================
# === STATE HANDLING ===
# ============================
if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False
if "top_results" not in st.session_state:
    st.session_state.top_results = None

# ============================
# === PROSES REKOMENDASI ===
# ============================

if st.button("Cari Lowongan"):
    if (
        job_preference == "Choose an option"
        or selected_type == "Choose an option"
        or selected_hours == "Choose an option"
    ):
        st.warning("Silakan pilih semua opsi terlebih dahulu.")
    else:
        user_input = f"{job_preference} {' '.join(selected_qualifications)} {selected_type} {selected_hours}"

        # Filter awal agar relevan
        filtered_df = df[(df["Type"] == selected_type) & (df["Work Hours"] == selected_hours)].copy()

        # === CEK JIKA DATA KOSONG ===
        if filtered_df.empty or "combined_features" not in filtered_df.columns:
            st.warning("Tidak ada data yang cocok dengan filter yang dipilih.")
            st.session_state.search_clicked = True
            st.session_state.top_results = None
        else:
            filtered_df["combined_features"] = filtered_df["combined_features"].fillna("").astype(str)

            if filtered_df["combined_features"].str.strip().eq("").all():
                st.warning("Data tidak memiliki deskripsi yang cukup untuk dihitung.")
                st.session_state.search_clicked = True
                st.session_state.top_results = None
            else:
                # === TF-IDF + COSINE SIMILARITY ===
                tfidf = TfidfVectorizer()
                tfidf_matrix = tfidf.fit_transform(filtered_df["combined_features"])
                user_vector = tfidf.transform([user_input])
                cosine_sim = cosine_similarity(user_vector, tfidf_matrix)

                filtered_df["similarity"] = cosine_sim[0]
                top_results = filtered_df.sort_values(by="similarity", ascending=False).head(5)

                st.session_state.search_clicked = True
                st.session_state.top_results = top_results if not top_results.empty else None

# ============================
# === HASIL REKOMENDASI ===
# ============================

if st.session_state.search_clicked:
    if st.session_state.top_results is not None:
        st.write(f"Menampilkan {len(st.session_state.top_results)} lowongan teratas:")

        display_df = st.session_state.top_results[[
            "Job Title", "Kota", "Work Hours", "Type", "Job Qualifications", "URL"
        ]].rename(columns={
            "Job Title": "Posisi",
            "Kota": "Kota",
            "Work Hours": "Jam Kerja",
            "Type": "Tipe",
            "Job Qualifications": "Kualifikasi",
            "URL": "Link"
        }).reset_index(drop=True)

        st.dataframe(display_df, use_container_width=True)

        folium_map = generate_map_with_counts(st.session_state.top_results)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_folium(folium_map, width=700, height=400)
    else:
        st.warning("Tidak ditemukan lowongan yang sesuai.")
