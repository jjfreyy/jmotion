# jMotion

jMotion adalah proyek _image-to-video_ berbasis Python yang menggunakan model **LTX-Video 0.9.8 13B Distilled** dari Hugging Face Diffusers.

Proyek ini mengubah satu gambar menjadi video berdasarkan instruksi gerakan melalui _positive prompt_ dan _negative prompt_. Proses inferensi menggunakan PyTorch dengan dukungan GPU NVIDIA CUDA serta optimasi penggunaan memori untuk menjalankan model berukuran besar.

## Fitur

- Menghasilkan video dari satu gambar referensi.
- Mengatur gerakan karakter menggunakan _positive prompt_.
- Mengurangi gerakan atau perubahan visual yang tidak diinginkan menggunakan _negative prompt_.
- Mempertahankan gambar referensi sebagai frame awal video.
- Menyesuaikan resolusi gambar secara otomatis.
- Menggunakan FP8 _layerwise casting_ untuk mengurangi penggunaan VRAM.
- Menggunakan _group offloading_ untuk memindahkan komponen model antara CPU dan GPU.
- Mengukur waktu generasi dan penggunaan VRAM.
- Mengekspor hasil generasi ke format MP4.

## Teknologi

| Teknologi              | Fungsi                                |
| ---------------------- | ------------------------------------- |
| Python 3.12.13         | Bahasa pemrograman                    |
| PyTorch                | Komputasi tensor dan inferensi GPU    |
| CUDA 13.4              | Akselerasi komputasi GPU              |
| Hugging Face Diffusers | Memuat pipeline dan menjalankan model |
| LTX-Video 0.9.8        | Model generasi video                  |
| Pillow                 | Membaca dan memproses gambar          |
| ImageIO                | Dukungan pemrosesan video             |
| python-dotenv          | Memuat konfigurasi dari file `.env`   |

## Struktur Proyek

```text
jmotion/
├── src/
│   └── main.ipynb
├── res/
│   ├── inputs/
│   │   └── img.jpg
│   └── outputs/
│       └── output.mp4
├── .env
├── r.txt
└── README.md
```

**Keterangan direktori:**

- `src/main.ipynb`: Notebook utama untuk konfigurasi model, pemrosesan gambar, dan generasi video.
- `res/inputs/`: Tempat menyimpan gambar referensi.
- `res/outputs/`: Tempat menyimpan video hasil generasi.
- `.env`: File konfigurasi variabel lingkungan.
- `r.txt`: Daftar dependensi Python.
- `README.md`: Dokumentasi proyek.

## Persyaratan Sistem

- Windows 11 64-bit.
- GPU NVIDIA dengan dukungan CUDA.
- Driver NVIDIA yang kompatibel dengan CUDA 13.4.
- Miniconda.
- Python 3.12.13.
- PyTorch dengan dukungan CUDA 13.4.

Model yang digunakan berukuran 13B parameter. Penggunaan GPU dengan VRAM 16 GB dimungkinkan dengan optimasi memori yang digunakan proyek ini, tetapi kebutuhan aktual dapat berbeda bergantung pada resolusi dan konfigurasi generasi.

## Instalasi

### 1. Instal Miniconda

Pastikan Miniconda sudah terinstal sebelum melanjutkan instalasi.

### 2. Buat environment Python

Buka Windows PowerShell, kemudian jalankan:

```powershell
conda create -n jmotion python=3.12.13
```

Aktifkan environment:

```powershell
conda activate jmotion
```

### 3. Instal PyTorch

Instal PyTorch dan TorchVision dengan dukungan CUDA 13.4:

```powershell
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu134
```

### 4. Instal dependensi

Instal seluruh dependensi yang tercantum dalam `r.txt`:

```powershell
pip install -r r.txt
```

### 5. Konfigurasi Hugging Face

Buat file `.env` di direktori utama proyek:

```env
HF_TOKEN=your_hf_token
```

Ganti `hf_token` dengan token Hugging Face milik Anda.

Token digunakan untuk autentikasi saat mengakses repositori model apabila diperlukan.

**Catatan:** Jangan membagikan token Hugging Face atau mengunggah file `.env` ke repositori publik.

## Model

JMotion menggunakan model berikut:

**Repositori:** [Lightricks/LTX-Video-0.9.8-13B-distilled](https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled)

Model akan diunduh dan dimuat melalui Hugging Face Diffusers ketika notebook dijalankan.

Lokasi cache Hugging Face pada konfigurasi notebook:

```python
pathHf = Path('D:/ai/hf')
os.environ['HF_HOME'] = str(pathHf)
```

Pastikan direktori tersebut tersedia dan memiliki ruang penyimpanan yang cukup.

## Konfigurasi Generasi

Parameter utama dapat diubah melalui bagian konfigurasi di `main.ipynb`.

| Parameter                |                                 Nilai awal | Keterangan                                                |
| ------------------------ | -----------------------------------------: | --------------------------------------------------------- |
| `REPO_ID`                | `Lightricks/LTX-Video-0.9.8-13B-distilled` | Repositori model                                          |
| `GUIDANCE_SCALE`         |                                      `1.0` | Kekuatan classifier-free guidance                         |
| `GUIDANCE_RESCALE`       |                                      `0.7` | Rescaling hasil guidance                                  |
| `DECODE_TIMESTEP`        |                                     `0.05` | Timestep untuk decoding VAE                               |
| `DECODE_NOISE_SCALE`     |                                    `0.025` | Tingkat noise tambahan saat decoding                      |
| `IMAGE_COND_NOISE_SCALE` |                                      `0.0` | Tingkat noise pada gambar referensi                       |
| `SEED`                   |                                       `42` | Seed untuk reproduksibilitas                              |
| `USE_FP8_LAYERWISE`      |                                     `True` | Mengaktifkan penyimpanan bobot transformer dalam FP8      |
| `USE_GROUP_OFFLOAD`      |                                     `True` | Mengaktifkan pemindahan komponen model antara CPU dan GPU |
| `USE_VAE_TILING`         |                                    `False` | Mengaktifkan VAE tiling                                   |
| `DTYPE`                  |                           `torch.bfloat16` | Tipe data komputasi                                       |
| `MAX_SIDE`               |                                      `832` | Batas maksimum sisi terpanjang gambar                     |
| `NUM_FRAMES`             |                                       `75` | Jumlah frame yang diminta                                 |
| `FPS`                    |                                       `24` | Jumlah frame yang diputar setiap detik                    |

### Sampling

Notebook menggunakan konfigurasi timestep berikut:

```python
TIMESTEPS = [
  1000, 993, 987, 981,
  975, 909, 725, 0.03,
]
```

Timestep menentukan urutan tahapan sampling yang digunakan model selama proses denoising.

### Optimasi Memori

**FP8 Layerwise Casting**

Menggunakan format `torch.float8_e4m3fn` untuk penyimpanan bobot transformer, sedangkan komputasi tetap menggunakan BF16.

**Group Offloading**

Memindahkan komponen model antara CPU dan GPU sesuai kebutuhan selama inferensi. Konfigurasi transformer menggunakan `leaf_level`, sedangkan text encoder menggunakan `block_level` dengan dua blok per kelompok.

**VAE Tiling**

Dapat diaktifkan untuk membantu mengurangi penggunaan VRAM saat proses decoding VAE.

## Menjalankan Proyek

### 1. Siapkan gambar

Letakkan gambar referensi di direktori:

```text
res/inputs/
```

Nama file gambar yang digunakan oleh notebook adalah:

```text
img.jpg
```

### 2. Buka notebook

Buka proyek menggunakan Visual Studio Code, kemudian buka:

```text
src/main.ipynb
```

Pastikan kernel Jupyter menggunakan environment `jmotion`.

### 3. Jalankan notebook

Jalankan seluruh sel notebook secara berurutan dari atas ke bawah.

Notebook akan menjalankan tahapan berikut:

1. Memuat konfigurasi dari `.env`.
2. Menentukan direktori proyek dan cache Hugging Face.
3. Menginisialisasi PyTorch dan CUDA.
4. Mengunduh atau memuat model LTX-Video.
5. Mengaktifkan optimasi memori.
6. Membaca dan menyesuaikan ukuran gambar referensi.
7. Menjalankan generasi video berdasarkan prompt.
8. Mengukur waktu generasi dan penggunaan VRAM.
9. Mengekspor hasil generasi menjadi video MP4.

## Mengatur Prompt

Notebook menyediakan dua parameter untuk mengendalikan gerakan video.

**Positive Prompt**

Menentukan gerakan, ekspresi, dan perilaku karakter yang ingin dihasilkan.

Contoh:

```python
PROMPT = (
  'Three adorable baby penguins dance happily together. '
  'They smile warmly, gently sway their bodies from side '
  'to side, flap their little wings rhythmically, and '
  'make tiny alternating foot taps. They occasionally '
  'tilt their heads playfully. All three dance in place '
  'with cheerful, synchronized movements. '
  'Preserve their original faces, eyes, beaks, body shapes, '
  'colors, and individual identities. '
  'Keep their original positions and spacing. '
  'Static camera, locked tripod, fixed framing and perspective. '
  'The background and lighting remain completely unchanged.'
)
```

**Negative Prompt**

Menentukan gerakan atau perubahan visual yang ingin dihindari.

Contoh:

```python
NEGATIVE_PROMPT = (
  'camera movement, zoom, pan, tilt, rotation, tracking, '
  'camera shake, reframing, perspective shift, '
  'walking away, moving forward, moving backward, '
  'position changes, jumping, falling, sliding, '
  'face distortion, identity change, morphing, '
  'deformed bodies, extra limbs, duplicated penguins, '
  'background changes, flickering, jittering, blur.'
)
```

Prompt dapat disesuaikan dengan karakter dan jenis gerakan yang diinginkan.

**Catatan:** Prompt merupakan instruksi untuk model, bukan batasan mutlak. Model tetap dapat menghasilkan perubahan identitas, gerakan kamera, atau gerakan yang tidak sepenuhnya sesuai instruksi.

## Hasil Generasi

Video hasil generasi secara otomatis disimpan pada:

```text
res/outputs/output.mp4
```

Format video yang digunakan adalah MP4.

Notebook juga menampilkan informasi berikut setelah generasi selesai:

- Waktu generasi.
- Penggunaan VRAM maksimum pada tensor PyTorch (_peak allocated_).
- Penggunaan memori maksimum yang dikelola _caching allocator_ (_peak reserved_).
- Jumlah frame yang berhasil dihasilkan.
- Resolusi video.
- Durasi video berdasarkan jumlah frame dan FPS.

Durasi video dihitung menggunakan:

```python
duration = len(videoFrames) / FPS
```

Jumlah frame aktual dapat berbeda dari nilai `NUM_FRAMES` yang diminta, bergantung pada pemrosesan pipeline.

## Catatan Tambahan

- Jalankan notebook dari direktori `src` agar path relatif `../res` mengarah ke direktori aset proyek.
- Pastikan gambar `img.jpg` tersedia sebelum menjalankan proses generasi.
- Generasi pertama dapat membutuhkan waktu lebih lama karena model harus diunduh terlebih dahulu.
- Penggunaan _group offloading_ dapat mengurangi kebutuhan VRAM, tetapi dapat menambah waktu inferensi akibat pemindahan data antara CPU dan GPU.
- Mengubah resolusi, jumlah frame, atau konfigurasi model dapat memengaruhi penggunaan VRAM dan waktu generasi.
- Seed yang sama membantu menghasilkan proses sampling yang lebih konsisten, tetapi tidak menjamin hasil identik pada setiap konfigurasi perangkat dan lingkungan.

## Lisensi

Model LTX-Video memiliki ketentuan lisensi tersendiri. Periksa lisensi dan ketentuan penggunaan pada repositori resmi [LTX-Video di Hugging Face](https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled) sebelum menggunakan atau mendistribusikan hasil proyek.
