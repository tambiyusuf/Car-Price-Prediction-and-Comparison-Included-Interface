from flask import Flask, request, jsonify, render_template
from fonksiyonlar import data_manipulation
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Formdan gelen verileri al
    marka = request.form['marka']
    seri = request.form['seri']
    ust_model = request.form['ust_model']
    alt_model = request.form['alt_model']
    yil = request.form['yil']
    sehir = request.form['sehir']
    kimden = request.form['kimden']
    renk = request.form['renk']
    yakit = request.form['yakit']
    kasa = request.form['kasa']
    km = request.form['km']
    motor_hacim = request.form['motor_hacim']
    hp = request.form['hp']
    kapi = request.form['kapi']
    vites = request.form['vites']
    takas = request.form['takas']
    agir_hasar = request.form['agir_hasar']
    garanti = request.form['garanti']
     # Fiyat değerini güvenli şekilde al
    fiyat = request.form.get('fiyat')  # Eğer yoksa None döner
    if fiyat:
        try:
            fiyat = int(fiyat)  # Fiyatı integer'a çevir
        except ValueError:
            return {"error": "Fiyat geçersiz bir değer içeriyor."}, 400  # Hatalı fiyat durumunu ele al

    # Örnek olarak alınan verileri birleştirip döndür
    response = {
        "marka": marka,
        "seri": seri,
        "ust_model": ust_model,
        "alt_model": alt_model,
        "yil": yil,
        "sehir": sehir,
        "kimden": kimden,
        "renk": renk,
        "yakit": yakit,
        "kasa": kasa,
        "km": km,
        "motor_hacim": motor_hacim,
        "hp": hp,
        "kapi": kapi,
        "vites": vites,
        "takas": takas,
        "agir_hasar": agir_hasar,
        "garanti": garanti
    }
    predicted_price = int(data_manipulation(response))
    
    if fiyat:
        if fiyat > predicted_price:
            status = "pahali"
            status_text = "PAHALI"
            fark= round(((fiyat - predicted_price) / predicted_price) * 100, 2)
        else:
            status = "uygun"
            status_text = "UYGUN"
            fark = round(((fiyat - predicted_price) / predicted_price) * 100, 2)
        print(predicted_price)
        return render_template('modelCevapFB.html', 
                           predicted_price=predicted_price, 
                           fiyat=fiyat, 
                           fark=abs(fark),
                           status=status,
                           status_text= status_text)
    else:
        print(response)
        print(predicted_price)
        return render_template('modelCevap.html', predicted_price= predicted_price)
    
    return str(predicted_price)

# Anasayfa rotası
@app.route('/')
def home():
    return render_template('anasayfa.html')
 
# Fiyat Biliyorum sayfası için rota
@app.route('/fiyatBiliyorum')
def fiyat_biliyorum():
    return render_template('fiyat_biliyorum.html')

# Fiyat İstiyorum sayfası için rota
@app.route('/fiyatIstiyorum')
def fiyat_istiyorum():
    return render_template('fiyat_istiyorum.html')



if __name__ == '__main__':
    app.run(debug=True)