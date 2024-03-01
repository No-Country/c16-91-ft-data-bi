import joblib

def predict(data):
    scaler = joblib.load('./streamlit_data/models/scaler.sav')
    norm_data = scaler.transform(data)
    clf = joblib.load('./streamlit_data/models/svm_model.sav')
    return clf.predict(norm_data)