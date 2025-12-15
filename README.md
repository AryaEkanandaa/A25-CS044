1. Deskripsi Singkat Project :
PrediX AI – Predictive Maintenance Copilot adalah website yang dirancang untuk memantau kondisi mesin secara real-time, memprediksi potensi kegagalan, serta mendeteksi anomali sejak dini berdasarkan data sensor. Sistem ini mengintegrasikan backend API, layanan machine learning, dan AI agent interaktif untuk membantu pengguna dalam menganalisis kondisi mesin, mengurangi downtime, dan mendukung pengambilan keputusan maintenance secara proaktif.

PrediX AI menggunakan Agentic AI untuk:
Mendeteksi anomali dari data sensor (getaran, temperatur, arus, dll) secara real-time.
Memberikan rekomendasi maintenance berbasis prediksi risiko peralatan.
Menjawab pertanyaan engineer menggunakan natural language query, misalnya “mesin mana yang paling berisiko minggu ini?”.
Mensimulasikan pembuatan tiket maintenance secara otomatis (dummy).

Tujuan utama proyek ini adalah mendukung engineer membuat keputusan data-driven, menurunkan risiko downtime tak terencana, dan menyajikan MVP web app yang mencakup dashboard kesehatan mesin serta interface chat agent. Keberhasilan proyek diukur dari akurasi model anomaly detection >70%, kemampuan agent menjawab query dengan benar, dan demo web app yang menampilkan dashboard dan interaksi chatbot.

2. Setup Environtment :

-FrontEnd
1. Pastikan Nodejs sudah terinstall (versi yang kami gunakan yaitu 22.19.0)
2. Masuk ke folder frontend pada terminal dengan perintah "cd frontend".
3. Jalankan perintah npm install
4. Terakhir, buat file .env pada folder frontend. Anda bisa melihat contoh isinya pada file .env.example.
   
-BackEnd :
1. Pastika Nodejs sudah terinstall (versi yang kami gunakan yaitu 22.19.0)
2. Masuk ke folder backend pada terminal dengan perintah "cd backend"
3. Jalankan perintah npm install
4. Terakhir, buat file .env pada folder backend. Anda bisa melihat contoh nya pada file .env.example

-Machine Learning :
1. Pastikan Python sudah terinstall (versi yang kami gunakan 3.11.9)
2. Masuk ke Folder machine learning pada terminal dengan perintah "cd ml-service".
3. Buat venv dengan perintah "python -m venv venv".
4. Kemudian aktifkan venv nya dengan perintah ".\venv\Scripts\activate".
5. Terakhir, Install library yang diperlukan dengan perintah " python -m pip install fastapi uvicorn scikit-learn numpy pandas requests xgboost".

3.Tautan Model ML : https://drive.google.com/drive/folders/1KfoNGtlG7G50trK2Q6XF4SVVyBgDg_9A?usp=sharing 

   
4.Cara Menjalankan Aplikasi :
   1. Buat seluruh environtment dan library yang diperlukan.
   2. Untuk Menjalankan server frontend. anda perlu masuk terlebih dahulu ke dalam folder frontend dengan perintah "cd frontend". setelah itu anda dapat menjalankan server dengan perintah "npm run dev ataupun npm run start".
   3. Untuk Menjalankan server backend. anda perlu masuk terlebih dahulu ke dalam folder backend dengan perintah "cd backend". setelah itu anda dapat menjalankan server backend dengan perintah "npm run dev ataupun npm run start".
   4. Untuk Menjalankan server ML. anda perlu masuk terlebih dahulu ke dalam folder ML dengan perintah "cd ml-service". setelah itu anda perlu mengaktifkan Venv dengan perintah "\venv\Scripts\activate". lalu untuk jalankan perintah "uvicorn server:app --reload --port 8001" untuk menjalankan server ML nya 
  

Arsitektur-AI-Agent : https://drive.google.com/file/d/1_1Km8ENOl8IA0p_K06jBN7U030ZU2X-g/view?usp=drive_link 
