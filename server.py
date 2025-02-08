from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from flask_cors import CORS
import pickle
from dotenv import load_dotenv
import os
import requests

# TODO load ENV
load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER="uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
BASE_API_URL = os.getenv("BASE_API_URL")

# TODO load Model
MODEL_PATH = os.path.join(os.getcwd(), "utils/model_no_vectorizer.keras")
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ["Negative", "Positive"]

# TODO load Stopwords
MAX_LENGTH = 32
nltk.download("stopwords")
stopwords_indonesia = stopwords.words("indonesian")

def standardize_func(sentence):
    """
    Removes a list of stopwords

    Args:
        sentence (tf.string): sentence to remove the stopwords from

    Returns:
        sentence (tf.string): lowercase sentence without the stopwords
    """
    stopwords = stopwords_indonesia
    sentence = tf.strings.lower(sentence)
    for word in stopwords:
        if word[0] == "":
            sentence =  tf.strings.regex_replace(sentence, rf"{word}\b", "")
        else:
            sentence = tf.strings.regex_replace(sentence, rf"\b{word}\b", "")

    sentence = tf.strings.regex_replace(sentence, r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~"]', "")
    return sentence

# TODO load Vectorizer
def load_vectorizer(file_path="vectorizer.pkl", standardize_func=None, MAX_LENGTH=32):
    with open(f"{os.getcwd()}/utils/{file_path}", "rb") as f:
        vocab = pickle.load(f)

    vectorizer = tf.keras.layers.TextVectorization(
        standardize=standardize_func,
        output_sequence_length=MAX_LENGTH
    )
    vectorizer.set_vocabulary(vocab)
    
    return vectorizer
vectorizer = load_vectorizer("vectorizer_vocab.pkl", standardize_func, MAX_LENGTH)


@app.route("/")
def hello_world():
    return "<h1>Welcome to SociaLens!</h1>"

@app.route("/api/model/predict/<int:restaurant_id>", methods=["POST"])
def predictions(restaurant_id):
    if "file" not in request.files:
        return jsonify({"message": "File not found"}), 404
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400
    
    if file and (file.filename.endswith(".csv")) or (file.filename.endswith(".xlsx")):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        try:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path, engine="openpyxl")
            
            #? nama ulasan tanggal [prediksi]
            ulasan_kosong = df["ulasan"].isna().sum()
            if ulasan_kosong > 1:
                return jsonify({"message": "Review null found"}), 400
            # TODO mengisi nama null ke string kosong 
            df["nama"] = df["nama"].fillna("")
            # TODO Format tanggal:
            df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
            
            #* mengecek semua format tanggal yang diberikan sudah sesuai
            if df["tanggal"].isna().sum() > 0:
                return jsonify({"message": "Invalid or missing date found"}), 400
            
            #* mengecek apakah ada tanggal yang terlewat
            start_date = df["tanggal"].min()
            end_date = df["tanggal"].max()
            expected_dates = pd.date_range(start=start_date, end=end_date)
            missing_date = set(expected_dates.date) - set(df["tanggal"].dt.date)
            
            #* return eror jika terdapat tanggal terlewat
            if missing_date:
                return jsonify({"message": "Missing date found"}), 400
            
            try:               
                # TODO Preprocessing & prediction
                print("tahap 1")

                # TODO standarisasi fungsi
                print("tahap 2")                
                test_sentences_tensor = tf.constant(df["ulasan"].values)
                test_texts_vectorized = vectorizer(test_sentences_tensor).numpy()

                # TODO prediksi model
                print("tahap 3")
                df["prediksi"] = model.predict(test_texts_vectorized)
                df["prediksi"] = df["prediksi"].apply(lambda p: class_names[int(p > 0.5)]) 
                df["restaurant_id"] = restaurant_id
                df["tanggal"] = df["tanggal"].dt.strftime("%Y-%m-%d")
                df = df.rename(columns={"tanggal": "time_review", "ulasan": "body", "nama": "username", "prediksi": "sentiment"})
                data_result = df.to_dict(orient="records")

                try: 
                    print("tahap 4")
                    print(BASE_API_URL)
                    express_response = requests.post(f"{BASE_API_URL}/api/reviews", json={"data": data_result})
                    return jsonify({"message": "Prediction successful", "express_response": express_response.json()}), 200

                except Exception as e:
                    return jsonify({"message": f"Error import data: {e}."}), 500
            except Exception as e:
                print(e)
                return jsonify({"message": "Error making prediction"}), 500
        except Exception as e:
            return jsonify({"message": "Error processing file"}), 500
    else:
        return jsonify({"message": "Invalid file format"}), 400 

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", port="7860")
