
import streamlit as st 
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element


#Sidebar
st.sidebar.title ("Dashboard")
app_mode = st.sidebar.selectbox("Selecty Page",["Home","About Project", "Prediction"])

#Main Page
if(app_mode == "Home"):
    st.header("FRUIT AND VEGETABLES RECOGNITION SYSTEM")
    image_path = "home.png"
    st.image(image_path)

#About Project
elif (app_mode == "About Project"):
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items : ")
    st.code("FRUITS : banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango")
    st.code("VEGETABLES : cucumber, carrot, capsium, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soybean, cauliflower, bell pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeno, ginger, garlic, peas, eggplant")
    st.subheader("CONTENT")
    st.text("This dataset contains three folders : ")
    st.text("1. Train (100 images each)")
    st.text("2. Test (10 images each)")
    st.text("3. Validation (10 images each)")
    
#Prediction Page
elif (app_mode=="Prediction"):
    st.header("MODEL PREDICTION")
    test_image = st.file_uploader("Choose an IMAGE : ")
    if(st.button("Show Image")):
        st.image(test_image,width=4, use_column_width = True)
    #Predicti button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        with open("label.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is Predicting it's a {}".format(label[result_index]))
            
        


