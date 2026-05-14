from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.config["DEBUG"] = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # sin esto no encuentran el archivo
CSV_PATH = os.path.join(BASE_DIR, 'Advertising.csv')
MODEL_PATH = os.path.join(BASE_DIR,'advertising.model')

def insert_data_csv(tv, radio, newspaper, sales):
    new_data = pd.DataFrame([[tv, radio, newspaper, sales]], 
                            columns=['TV', 'radio', 'newspaper', 'sales'])
    
    if not os.path.exists(CSV_PATH):
        new_data.to_csv(CSV_PATH, index=False)
    else:
        new_data.to_csv(CSV_PATH, mode='a', header=False, index=False)

def train_model():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("El archivo CSV no existe. Ingrese datos primero.")
    
    df = pd.read_csv(CSV_PATH)
    X = df[['TV', 'radio', 'newspaper']]
    y = df['sales']
    
    model = LinearRegression()
    model.fit(X, y)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

@app.route('/')
def home():
    return "<h1>API para predecir ventas de publicidad</h1><p>Acceda a la documentación de la API en /docu</p>"

@app.route('/docu')
def docu():
    return "<p>/predecir?tv=tv&radio=radio&newspaper=newspaper</p><p>/reentrenar (POST)</p><p>/ingresar (POST)</p><p>/datos</p>"

@app.route('/datos', methods=['GET'])
def devolver_datos():
    try:
        if not os.path.exists(CSV_PATH):
            return jsonify({"error": "El archivo CSV aún no existe"}), 404
        # Leer el CSV
        df = pd.read_csv(CSV_PATH)
        data = df.to_dict(orient='records')
        return jsonify(data), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predecir', methods=['GET'])
def predecir():
    try:
        tv = request.args.get('tv', type=float)
        radio = request.args.get('radio', type=float)
        newspaper = request.args.get('newspaper', type=float)
        
        if None in [tv, radio, newspaper]:
            return jsonify({"error": "Faltan parámetros"}), 400

        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        prediction = model.predict([[tv, radio, newspaper]])
        return jsonify({'prediction': float(prediction[0])})
    
    except FileNotFoundError:
        return jsonify({"error": "Modelo no disponible"}), 500

@app.route('/ingresar', methods=['GET', 'POST'])
def ingest():
    tv = request.args.get('tv', type=float)
    radio = request.args.get('radio', type=float)
    newspaper = request.args.get('newspaper', type=float)
    sales = request.args.get('sales', type=float)
    
    if None in [tv, radio, newspaper, sales]:
        data = request.get_json(silent=True)
        if data:
            tv = data.get('tv')
            radio = data.get('radio')
            newspaper = data.get('newspaper')
            sales = data.get('sales')

    if None in [tv, radio, newspaper, sales]:
        return jsonify({"error": "Datos incompletos, me falta alguno de los datos anteriores,revisalos"}), 400
    
    insert_data_csv(tv, radio, newspaper, sales)
    return jsonify({"status": "Registro insertado correctamente"}), 201

@app.route('/reentrenar', methods=['GET','POST'])
def reentrena():
    try:
        train_model()
        return jsonify({"status": "Modelo actualizado con los datos del CSV"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()