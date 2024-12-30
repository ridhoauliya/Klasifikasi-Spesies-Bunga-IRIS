import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Memuat dataset
def load_data():
    return pd.read_csv("IRIS.csv")

# Sidebar
st.sidebar.title("Klasifikasi Dataset IRIS")
model_terpilih = st.sidebar.selectbox("Pilih Model", ["Random Forest"])

# Aplikasi Utama
st.title("Analisis dan Klasifikasi Dataset IRIS")

# Memuat data
data = load_data()
st.write("### Pratinjau Dataset", data.head())

# Preprocessing
def preprocess_data(data):
    data = data.dropna()  #1 Menghapus nilai kosong
    label_encoder = LabelEncoder()
    data["species"] = label_encoder.fit_transform(data["species"])  #2 Mengubah label menjadi numerik
    scaler = StandardScaler()
    fitur = data.drop("species", axis=1)  # Memisahkan fitur
    fitur_terstandarisasi = scaler.fit_transform(fitur)  #3 Menstandarkan fitur
    data_terproses = pd.DataFrame(fitur_terstandarisasi, columns=fitur.columns)
    data_terproses["species"] = data["species"]  # Menambahkan label yang telah dienkode
    return data_terproses, label_encoder

#4 Memproses data
data_terproses, label_encoder = preprocess_data(data)
st.write("### Dataset yang Telah Diproses", data_terproses.head())

# Analisis Data Eksploratif
st.header("Analisis Data Eksploratif")
st.subheader("Ringkasan Statistik")
st.write(data_terproses.describe())

# Visualisasi
st.subheader("Distribusi Fitur")
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.histplot(data_terproses["sepal_length"], kde=True, ax=axs[0, 0])
axs[0, 0].set_title("Panjang Sepal")
sns.histplot(data_terproses["sepal_width"], kde=True, ax=axs[0, 1])
axs[0, 1].set_title("Lebar Sepal")
sns.histplot(data_terproses["petal_length"], kde=True, ax=axs[1, 0])
axs[1, 0].set_title("Panjang Petal")
sns.histplot(data_terproses["petal_width"], kde=True, ax=axs[1, 1])
axs[1, 1].set_title("Lebar Petal")
st.pyplot(fig)

st.subheader("Pairplot")
st.pyplot(sns.pairplot(data, hue="species"))

# Heatmap
st.subheader("Heatmap Korelasi")
plt.figure(figsize=(8, 6))
sns.heatmap(data_terproses.corr(), annot=True, cmap="coolwarm")
st.pyplot(plt)

# Menyiapkan fitur dan label
X = data_terproses.drop("species", axis=1)
y = data_terproses["species"]

# Membagi data train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pelatihan Model
st.header("Pelatihan dan Evaluasi Model")
if model_terpilih == "Random Forest":
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrik Evaluasi
    akurasi = accuracy_score(y_test, y_pred)
    st.write(f"### Akurasi: {akurasi:.2f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    st.pyplot(plt)

    st.subheader("Laporan Klasifikasi")
    laporan = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    st.text(laporan)

st.sidebar.write("---")
st.sidebar.write("Kelompok 2.0")
st.sidebar.write("Fauzan Ar muzadi")
st.sidebar.write("Ridho Auliya Fatwa")