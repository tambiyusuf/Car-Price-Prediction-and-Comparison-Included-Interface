import pandas as pd 
import json 
import numpy
import joblib
from tensorflow.keras.models import load_model
import os 

# Şehir grupları tanımlanacak ve dönüşümleri yapılacak. + 
# Garanti, Takas, Ağir Hasar, Vites verilerine label encoder uygulanacak + 
# BMW 1 Serisi ve 4 Serisi olarak gelen verilerden 'Seri'metni kaldırılacak. + 
# Kasa verisinden gelen kasa tipi ve çekiş verisi '-' na göre split edilecek. + 
"""
hot encoder uygulanan veriler;

-> marka_{gelen veri} şeklinde yapıldı
-> seri_{gelen veri} şeklinde yapıldı
-> alt_model_{gelen veri} şeklinde yapıldı
-> ust_model_{gelen veri} şeklinde yapıldı
-> kasa_{gelen veri}
-> cekis_{gelen veri}

"""
import pandas as pd 
def columns_fitting(df):
    pre_model_columns = ['yil',
 'vites',
 'km',
 'hp',
 'kapi',
 'garanti',
 'agirHasar',
 'takas',
 'motorHacim',
 'sehir',
 'marka_Alfa Romeo',
 'marka_Audi',
 'marka_BMW',
 'marka_Chevrolet',
 'marka_Dodge',
 'marka_Fiat',
 'marka_Ford',
 'seri_1 si',
 'seri_4 si',
 'seri_A4',
 'seri_Cruze',
 'seri_Egea',
 'seri_Escort',
 'seri_Fiesta',
 'seri_Focus',
 'seri_Kuga',
 'seri_Linea',
 'seri_Mondeo',
 'seri_Nitro',
 'seri_Puma',
 'seri_Ranger',
 'seri_Tonale',
 'seri_X1',
 'seri_X3',
 'seri_X5',
 'seri_iX',
 'kasa_Cabrio',
 'kasa_Coupe',
 'kasa_Crossover',
 'kasa_Hatchback',
 'kasa_Pickup',
 'kasa_SUV',
 'kasa_Sedan',
 'kasa_Station Wagon',
 'renk_Bej',
 'renk_Beyaz',
 'renk_Bordo',
 'renk_Füme',
 'renk_Gri',
 'renk_Gümüş Gri',
 'renk_Kahverengi',
 'renk_Kırmızı',
 'renk_Lacivert',
 'renk_Mavi',
 'renk_Mor',
 'renk_Sarı',
 'renk_Siyah',
 'renk_Turuncu',
 'renk_Yeşil',
 'renk_Şampanya',
 'cekis_4x2 (Arkadan İtişli)',
 'cekis_4x2 (Önden li)',
 'cekis_4x4',
 'cekis_Arkadan İtiş',
 'cekis_Önden',
 'ust_model_0',
 'ust_model_1.0 EcoBoost',
 'ust_model_1.3 Multijet',
 'ust_model_1.5 EcoBlue',
 'ust_model_1.5 EcoBoost',
 'ust_model_1.5 Hybrid',
 'ust_model_1.5 TDCI',
 'ust_model_1.5 TDCi',
 'ust_model_1.6',
 'ust_model_1.6 CLX',
 'ust_model_1.6 D',
 'ust_model_1.6 EcoBoost',
 'ust_model_116i',
 'ust_model_16d sDrive',
 'ust_model_16i sDrive',
 'ust_model_18i sDrive',
 'ust_model_2.0 EcoBlue',
 'ust_model_2.0 Ghia',
 'ust_model_2.2 TDCi',
 'ust_model_2.5 TDCi',
 'ust_model_2.8 CRD',
 'ust_model_20d xDrive',
 'ust_model_20i sDrive',
 'ust_model_25d xDrive',
 'ust_model_3.2 TDCi',
 'ust_model_30d',
 'ust_model_418i Gran Coupe',
 'ust_model_420d',
 'ust_model_44',
 'ust_model_A4 Sedan',
 'ust_model_xDrive40',
 'ust_model_xDrive50',
 'alt_model_0',
 'alt_model_1.4 TDCi Comfort',
 'alt_model_2.0 TDI',
 'alt_model_2.0d xDrive',
 'alt_model_4x4 Wildtrak',
 'alt_model_Active Plus',
 'alt_model_Comfort',
 'alt_model_Easy',
 'alt_model_Exclusive',
 'alt_model_First Edition Sport',
 'alt_model_Hybrid ST-Line',
 'alt_model_LS',
 'alt_model_M Sport',
 'alt_model_Pop',
 'alt_model_Premium',
 'alt_model_Prestige',
 'alt_model_ST Line',
 'alt_model_STD',
 'alt_model_SXT',
 'alt_model_Selective',
 'alt_model_Standart',
 'alt_model_Style',
 'alt_model_Ti',
 'alt_model_Titanium',
 'alt_model_Trend X',
 'alt_model_Veloce',
 'alt_model_Wild Trak',
 'alt_model_X Line',
 'alt_model_XL',
 'alt_model_XLT',
 'kimden_Galeriden',
 'kimden_Sahibinden',
 'kimden_Yetkili Bayiden',
 'yakit_Benzin',
 'yakit_Benzin & LPG',
 'yakit_Dizel',
 'yakit_Elektrik',
 'yakit_Hybrid']
    # Eksik sütunları belirle
    pre_model_set = set(pre_model_columns)
    df_set = set(df.columns)
    missing_columns = pre_model_set - df_set

    # Eksik sütunları toplu olarak ekle
    if missing_columns:
        missing_df = pd.DataFrame(0, index=df.index, columns=list(missing_columns))
        df = pd.concat([df, missing_df], axis=1)

    # Sütun sırasını düzenle
    df = df[pre_model_columns]

    return df

def data_manipulation(response):
    df = pd.DataFrame([response])


    if 'fiyat' in df.columns:
        fiyat = df["fiyat"]

    # Label Encoder uygulanırken baz alınan şehir grupları
    city_group = ['Adıyaman ', 'Bitlis ', 'İstanbul ', 'Muş ', 'Samsun ', 'Kocaeli ', 'Erzincan ', 'İzmir ', 'Muğla ', 'Gaziantep ', 'Mersin ', 'Afyonkarahisar ', 'Mardin ', 'Kayseri ', 'Bursa ', 'Siirt ', 'Eskişehir ', 'Denizli ', 'Şanlıurfa ']
    city_group2 =['Sakarya ', 'Balıkesir ', 'Kahramanmaraş ', 'Hatay ', 'Bilecik ', 'Tunceli ', 'Manisa ', 'Aksaray ', 'Kütahya ', 'Zonguldak ', 'Ağrı ', 'Nevşehir ', 'Tokat ', 'Kırklareli ', 'Tekirdağ ', 'Niğde ', 'Konya ', 'Sivas ', 'Batman ', 'Ankara ']
    city_group3 =  ['Karaman ', 'Burdur ', 'Yalova ', 'Düzce ', 'Bolu ', 'Kırşehir ', 'Osmaniye ', 'Van ', 'Giresun ', 'Iğdır ', 'Aydın ', 'Edirne ', 'Sinop ', 'Ordu ', 'Şırnak ', 'Diyarbakır ', 'Karabük ', 'Kilis ', 'Erzurum ']
    city_group4 = ['Isparta ', 'Yozgat ', 'Elazığ ', 'Trabzon ', 'Çorum ', 'Çanakkale ', 'Uşak ', 'Antalya ', 'Bartın ', 'Kastamonu ', 'Çankırı ', 'Amasya ', 'Rize ', 'Kars ', 'Adana ', 'Kırıkkale ', 'Malatya ', 'Bingöl ', 'Hakkari ', 'Artvin ']
    # Gruplara göre şehirlerin kodlanması
    sehir_kodlama = {}
    for sehir in city_group:
        sehir_kodlama[sehir] = 4  # 1. grup -> 4
    for sehir in city_group2:
        sehir_kodlama[sehir] = 3  # 2. grup -> 3
    for sehir in city_group3:
        sehir_kodlama[sehir] = 2  # 3. grup -> 2
    for sehir in city_group4:
        sehir_kodlama[sehir] = 1  # 4. grup -> 1
    # Eğitim ve test veri setlerindeki şehir kolonlarını kodlama
    df['sehir'] = df['sehir'].map(sehir_kodlama)

    #------------------------------------------------------------------

    garanti_kodlama = {'Var': 1, 'Yok': 0}
    vites_kodlama = {'Manuel': 0, 'Otomatik': 1}
    takas_kodlama = {'Evet': 1, 'Hayır': 0}
    agirHasar_kodlama = {'Evet': 0, 'Hayır': 1}
    # fdf üzerindeki kolonlara kodlama işlemi
    df['garanti'] = df['garanti'].map(garanti_kodlama)
    df['vites'] = df['vites'].map(vites_kodlama)
    df['takas'] = df['takas'].map(takas_kodlama)
    df['agir_hasar'] = df['agir_hasar'].map(agirHasar_kodlama)

    #-------------------------------------------------------------------
    # Eğer BMW 1 Serisi ya da 4 Serisi gelirse bunlar ('1 si','4 si') olarak değiştirilecek çünkü model eğitimin de bu yapıldı.
    if df['seri'].str.contains('Seri').any():
        df['seri'] = df['seri'].str.replace('Seri', '').str.strip()
        if df['ust_model'].str.contains('Seri').any():
            df['ust_model'] = df['ust_model'].str.replace('Seri', '').str.strip()
    #-----------------------------------------------------------------
    # kasa ve cekis bilgisi beraber gelir, kullanıcıdan. Her araçta cekis bilgisi olmayabilir de,
    # eğer gelen veriyi ayır 'cekis' kolonuna veriyi ata dersek none veride atanabilir bu nedenle eğer varsa cekis bilgisi eğitime
    # giden veri setinde olduğu gibi, kodlanarak eklenir.
    for index, row in df.iterrows():
        if '-' in row['kasa']:
           kasa, cekis = row['kasa'].split('-')
           df.at[index, 'kasa'] = kasa.strip()
           cekis_column_name = f'cekis_{cekis.strip()}'
           df[cekis_column_name] = 0  # Yeni kolonu sıfırla
           df.at[index, cekis_column_name] = 1  # Sadece ilgili satıra 1 ata

    #-------------------------------------------------------------------
    # OHE
    target_columns = ['marka', 'seri', 'ust_model', 'alt_model', 'kasa', 'kimden', 'renk', 'yakit']
    for column in target_columns:
        for value in df[column]:
            new_column_name = f"{column}_{value}"
            df[new_column_name] = 1
        # Eski sütunu kaldır
        df.drop(column, axis=1, inplace=True)

    df = df.rename(columns={'agir_hasar': 'agirHasar', 'motor_hacim': 'motorHacim'})
    df = df.astype('float64')

    #---------------------------
    
    df = columns_fitting(df)

    #________________Using Trained Model_________________
    # Scaler'ı yüklemek
    project_root = os.path.dirname(os.path.abspath(__file__))
    scaler_path =  os.path.join(project_root,"model", "scaler.pkl")
    model_path = os.path.join(project_root,"model", "last_model.keras")

    scaler = joblib.load(scaler_path)
    model = load_model(model_path)

    # 3. Kullanıcı verisini alın (df'den örnek alalım, burada ilk satır örnek olarak alınıyor)
    user_input = df.iloc[0].values  # df'nin ilk satırını alıyoruz (veya manuel veri girişi yapabilirsiniz)

    # 4. Kullanıcı verisini uygun şekle getirin (reshape)
    user_input = user_input.reshape(1, -1)  # (1, n_features) şeklinde olacak

    # 5. Veriyi normalleştirin (scaler ile)
    user_input_scaled = scaler.transform(user_input)

    # 6. Model ile tahmin yapma
    prediction = model.predict(user_input_scaled)

    return prediction[0][0]


