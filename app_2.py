import streamlit as st
import pickle
from datetime import datetime

startTime = datetime.now()

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model_2.sv"
model = pickle.load(open(filename, 'rb'))

sex_d = {0: "Kobieta", 1: "Człowiek"}
chest_pain_d = {1: "ATA", 2: "NAP", 0: "ASY", 3: "TA"}
resting_ecg_d = {1: "Normal", 2: "ST", 0: "LVH"}
exercise_angina_d = {1: "Tak", 0: "Nie"}
st_slope_d = {2: "Up", 1: "Flat", 0: "Down"}
fastingBS_d = {0: "Nie", 1: "YEs"}


def main():
    st.set_page_config(page_title="Heart Disease Prediction App")
    overview = st.container()
    column = st.container()
    prediction = st.container()

    st.image("img.jpg")

    with overview:
        st.title("Heart Disease Prediction Zadanie2_s22599")

    with column:
        age_slider = st.slider("Wiek", value=28, min_value=28, max_value=77)
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])
        chest_pain_radio = st.radio("Ból w klatce piersiowej", list(chest_pain_d.keys()), format_func=lambda x: chest_pain_d[x])
        restingBp_slider = st.slider("RestingBP", value=1, min_value=1, max_value=200)
        cholesterol_slider = st.slider("Cholesterol", value=1, min_value=1, max_value=603)
        fastingBS_radio = st.radio("FastingBS", list(fastingBS_d.keys()), format_func=lambda x: fastingBS_d[x])
        resting_ecg_radio = st.radio("RestingECG", list(resting_ecg_d.keys()), format_func=lambda x: resting_ecg_d[x])
        maxHR_slider = st.slider("MaxHR", value=60, min_value=60, max_value=202)
        exercise_angina_radio = st.radio("ExerciseAngina", list(exercise_angina_d.keys()), format_func=lambda x: exercise_angina_d[x])
        oldpeak_slider = st.slider("Oldpeak", value=-2.6, min_value=-2.6, max_value=6.2)
        st_slope_radio = st.radio("ST_Slope", list(st_slope_d.keys()), format_func=lambda x: st_slope_d[x])

    data = [[age_slider, sex_radio, chest_pain_radio, restingBp_slider, cholesterol_slider, fastingBS_radio,
             resting_ecg_radio, maxHR_slider, exercise_angina_radio, oldpeak_slider, st_slope_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Model przewiduje, że u pacjenta prawdopodobnie wystąpi choroba serca?")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()
