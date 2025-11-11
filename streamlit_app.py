import streamlit as st
import pandas as pd
import json
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Atur layout halaman Streamlit
st.set_page_config(layout="wide")
st.title("Rekomendasi Lowongan Pekerjaan Pengolahan Data")

# Load data lowongan pekerjaan
df = pd.read_csv('Data/Data_Clean.csv')

# Bersihkan data: hilangkan baris dengan nilai kosong pada kolom penting
df.dropna(subset=["Job Title", "Job Qualifications", "Type", "Work Hours", "Kota", "URL"], inplace=True)

# Isi NaN di kualifikasi jika ada
df["Job Qualifications"] = df["Job Qualifications"].fillna("")

# Gabungkan fitur-fitur untuk proses TF-IDF
df["combined_features"] = (
    df["Job Qualifications"] + " " + df["Job Qualifications"] + " " + df["Job Title"] + " " + df["Type"] + " " + df["Work Hours"]
)

# Load data lokasi kota dari file JSON
with open("Data/location.json") as f:
    location_data = json.load(f)

# Fungsi untuk parsing koordinat dari format "POINT (lng lat)"
def parse_point(geo_str):
    coord_str = geo_str.replace("POINT (", "").replace(")", "")
    lng_str, lat_str = coord_str.split()
    return float(lat_str), float(lng_str)

# Fungsi untuk membuat peta dengan marker cluster jumlah lowongan per kota
def generate_map_with_counts(results):
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

# Form input dari pengguna
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

# Pilihan kualifikasi (bisa lebih dari satu)
all_qualifications = sorted(df["Job Qualifications"].str.split(",").explode().str.strip().dropna().unique())
selected_qualifications = st.multiselect("Kualifikasi Anda", all_qualifications)

# Inisialisasi state Streamlit
if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False
if "top_results" not in st.session_state:
    st.session_state.top_results = None

# Tombol untuk memulai pencarian lowongan
if st.button("Cari Lowongan"):
    # Validasi input: pastikan semua opsi telah dipilih
    if (
        job_preference == "Choose an option"
        or selected_type == "Choose an option"
        or selected_hours == "Choose an option"
    ):
        st.warning("Silakan pilih semua opsi terlebih dahulu.")
    else:
        # Buat input pengguna sebagai string gabungan untuk TF-IDF
        user_input = f"{job_preference} {' '.join(selected_qualifications)} {selected_type} {selected_hours}"

        # FILTERING AWAL berdasarkan tipe & jam kerja agar hasil lebih relevan
        filtered_df = df[(df["Type"] == selected_type) & (df["Work Hours"] == selected_hours)].copy()

        # Hitung TF-IDF hanya pada data yang lolos filter
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(filtered_df["combined_features"])
        user_vector = tfidf.transform([user_input])
        cosine_sim = cosine_similarity(user_vector, tfidf_matrix)

        # Tambahkan kolom skor similarity dan ambil top 5 teratas
        filtered_df["similarity"] = cosine_sim[0]
        top_results = filtered_df.sort_values(by="similarity", ascending=False).head(5)

        # Simpan hasil di session
        st.session_state.search_clicked = True
        st.session_state.top_results = top_results if not top_results.empty else None

# Tampilkan hasil jika pencarian telah dilakukan
if st.session_state.search_clicked:
    if st.session_state.top_results is not None:
        st.write(f"Menampilkan {len(st.session_state.top_results)} lowongan teratas:")

        # Tampilkan tabel hasil lowongan yang direkomendasikan
        display_df = st.session_state.top_results[[
            "Company Name", "Job Title", "Kota", "Work Hours", "Type", "Job Qualifications", "URL"
        ]].rename(columns={
            "Company Name": "Nama Perusahaan",
            "Job Title": "Posisi",
            "Kota": "Kota",
            "Type": "Tipe",
            "Work Hours": "Jam Kerja",
            "Job Qualifications": "Kualifikasi",
            "URL": "Link Access"
        }).reset_index(drop=True)

        st.dataframe(display_df, use_container_width=True)

        # Tampilkan peta lowongan per kota
        folium_map = generate_map_with_counts(st.session_state.top_results)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_folium(folium_map, width=700, height=400)
    else:
        st.warning("Tidak ditemukan lowongan yang sesuai.")
