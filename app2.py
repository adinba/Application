################ Application

### Importation
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import cv2
import heapq
import pandas as pd
import tensorflow as tf
from sklearn.cluster import DBSCAN
import pickle
from annotated_text import annotated_text
from streamlit_drawable_canvas import st_canvas
import json

### Chargement model et label
def importation_model_et_label(repertoire):
    model = tf.keras.models.load_model(repertoire + '/model.hd5')
    labels = []
    with open(repertoire + '/labels', 'rb') as temp:
        labels = pickle.load(temp)
    return model,labels

model,liste_h = importation_model_et_label("./model")


### Récupération du fichier json contenant les différentes langues
fichier_json = open('./language.json', 'r', encoding="utf-8")

with fichier_json as fichier:
   messages = json.load(fichier)
   

### Par défault la langue est l'anglais
langue = "EN"


##### App
st.set_page_config(page_title="App", page_icon="⚕️", layout="centered", initial_sidebar_state="expanded")

logo = st.sidebar.image(cv2.resize(np.array(Image.open("images/logo.png")),(80,100)))

html_temp = """ 
    <img src="images/logo.png" alt = "LOGO"> 
    """
st.markdown(html_temp, unsafe_allow_html = True) 

langue = str(st.sidebar.selectbox(messages[langue]["language_choice"], options=["EN","FR","ESP"]))


html_temp = """ 
    <div style ="background-color:#af81c2;padding:13px"> 
    <h1 style ="color:black;text-align:center;">OCR-PT-CT</h1> 
    </div> 
    """
      
st.markdown(html_temp, unsafe_allow_html = True) 
#st.subheader('Author')

st.set_option('deprecation.showfileUploaderEncoding', False)

title_alignment ="""
    <div style = "background-color:#b897c6">
    <h1 style = "color:blac;text-align:center;">Option de paramétrage</h1>
    </div>
    """
st.sidebar.markdown(title_alignment, unsafe_allow_html=True)

img_file = st.sidebar.file_uploader(label=messages[langue]["1"], type=['png', 'jpg'])
box_color = st.sidebar.color_picker(label=messages[langue]["countours"], value='#0000FF')
aspect_choice = st.sidebar.selectbox(label=messages[langue]["ratio"], options=["1:1", "16:9", "4:3", "2:3", "Libre"])
aspect_dict = {"1:1": (1, 1),"16:9": (16, 9),"4:3": (4, 3),"2:3": (2, 3),"Libre": None}
aspect_ratio = aspect_dict[aspect_choice]

option_choice = st.sidebar.radio(label="Options", options=["Recherche des hyéroglyfes", "Décodage", "Traduction","Reconnaissance","Dessin"])

save = st.sidebar.button('Sauver le résultat')

### Couleur hexadécimale en RGB
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


if img_file:
    col = hex_to_rgb(box_color)
    img = Image.open(img_file)
    cropped_img = st_cropper(img, realtime_update=True, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    _ = cropped_img.thumbnail((1500,1500))
    stock_cropped = st.image(cropped_img)
    
    if option_choice == "Recherche des hyéroglyfes":
        array = np.array(cropped_img)
        stock_cropped.empty()
        try:
            gray = cv2.cvtColor(array.copy(), cv2.COLOR_BGR2GRAY)
        except:
            gray = array.copy()
        
        thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (numLabels, _, stats, centroids) = output
        
        
        for i in range(0, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
                
            Keep_area = area < 30000 and area > 10
            Keep_w_h = w <80 and h <80 and w > 3 and h> 3
                        
            if all((Keep_area,Keep_w_h)):
                cv2.rectangle(array, (x, y), (x + w, y + h), col, 2)
                
        array = cv2.resize(array, (1500,1000),interpolation=cv2.INTER_AREA)
        st.image(array, caption='')
    
    if option_choice == "Décodage":
        
        array = np.array(cropped_img)
        img = array.copy()
        stock_cropped.empty()
        try:
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        except:
            gray = array
        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
              
        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (numLabels, _, stats, centroids) = output
        
        frame = pd.DataFrame(columns = ["x"],index = list(range(numLabels)))
            
        for i in range(0, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
                
            Keep_area = area < 30000 and area > 10
            Keep_w_h = w <80 and h <80 and w > 3 and h> 3

            if all((Keep_area,Keep_w_h)):
                frame.iloc[i,] = cX
                
        # Recherche des indices contenant des valeurs manquantes afin de faire les groupes en utilisants les symboles s'apparatant a etre des
        # hyéroglyfes
        index_with_nan = frame.index[frame.isnull().any(axis=1)]
        frame.drop(index_with_nan,0, inplace=True)
        centroids = pd.DataFrame(centroids,columns = ["x","y"])
        centroids.drop(index_with_nan,0, inplace=True)

        # Clustering
        frame = np.array(frame)
        db = DBSCAN(eps=20, min_samples=4).fit(frame)
        labels = db.labels_

        # Data frame résultant
        groupe = pd.DataFrame(np.array((labels,centroids["x"],centroids["y"])).transpose(),columns = ["groupe","x","y"])
        
        # Group by groupe
        resultat = groupe.groupby(by='groupe', as_index=False).agg({'x': pd.Series.nunique})

        # Les éléments noté -1 ne sont dans aucun groupe
        idx = []
        for i in range(groupe.shape[0]):
            if groupe["groupe"][i]  == -1:
                idx.append(i)
        groupe = groupe.drop(idx)
        
        # On récupére la moyenne des absicces de chaques groupes.
        points = groupe.groupby("groupe").agg({"x":"mean","y":["max","min"]})
        moyenne_x_grp = points["x"]["mean"]

        # On tri les groupes par rapport a leur absicces
        sort_moyenne_x_grp = np.sort(moyenne_x_grp)


        ecart_a_x = 30
        (numLabels, _, stats, centroids) = output

        # On récupère les éléments selon nos groupes clusterisé
        elements = []
        for i in range(0, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]

            Keep_area = area < 3000 and area > 15
            Keep_w_h = w <60 and h < 60 and w > 1 and h>1
    
            if all((Keep_area,Keep_w_h)):
                if np.min(np.abs(sort_moyenne_x_grp - x)) < ecart_a_x: # Vérfication de la bonne colonne
                    groupe = np.argmin(np.abs(sort_moyenne_x_grp - x)) # Détermination du groupe du symbole
                    
                    # Prédiction, sauvegarde du résultat et écriture sur l'image
                    temp = img[y:(y+h),x:(x+w)] 
                    temp = cv2.resize(temp, (100,100),interpolation=cv2.INTER_AREA)
                    img_data = np.expand_dims(temp, axis=0)
                    vecteur = model.predict(img_data)
                    res = liste_h[np.argmax(vecteur)]
                    elements.append({"area":area,"x":x,"y":y,"w":w,"h":h,"cX":cX,"cY":cY,"groupe":groupe,"pred":res})
                    cv2.putText(array,str(res),(x + w, y + h),cv2.FONT_HERSHEY_TRIPLEX , 0.5, col,thickness = 1)

        # Affichage de l'image
        array = cv2.resize(array, (1500,1000),interpolation=cv2.INTER_AREA)
        st.image(array, caption='')
        
        # Un data frame pour chaque colonne
        frame = pd.DataFrame(elements)
        groupe_by_frame = frame.groupby("groupe")
        liste_colonne_elem = []
        for i in range(sort_moyenne_x_grp.shape[0]):
            liste_colonne_elem.append(groupe_by_frame.get_group(i))
    
        def juxtaposition(elem1,elem2):
            if elem2["cY"] > elem1["y"] and elem2["cY"] < (elem1["y"] + elem1["h"]):
                return True
            return False



        def superposition(elem1,elem2,max_h,elem3 = None):
            if not elem3 is None:
                h = abs(elem3["y"] - (elem1["y"] + elem1["h"]))
            else:
                h = abs(elem2["y"] - (elem1["y"] + elem1["h"]))

            if h < max_h:
                return True
            return False

        def encodage(liste_elem):
            result = ""
            it = 0
            Q3_h = liste_elem["h"].quantile(0.75)  
            while it < (liste_elem.shape[0] - 2):
                if juxtaposition(liste_elem.iloc[it],liste_elem.iloc[it+1]):
                    result += ( "-" + str(liste_elem.iloc[it]["pred"]) + "*" + str(liste_elem.iloc[it + 1]["pred"]))
                    it += 1
                elif juxtaposition(liste_elem.iloc[it+1],liste_elem.iloc[it]) and juxtaposition(liste_elem.iloc[it+1],liste_elem.iloc[it+2]):
                    result += ( "-" + "(" +str(liste_elem.iloc[it]["pred"]) + ":" + str(liste_elem.iloc[it + 2]["pred"]) + ")*" + str(liste_elem.iloc[it+1]["pred"]))
                    it += 2
                elif superposition(liste_elem.iloc[it],liste_elem.iloc[it+1],Q3_h,elem3 = liste_elem.iloc[it+2]):
                    if juxtaposition(liste_elem.iloc[it +1],liste_elem.iloc[it + 2]):
                        result += ("-" + str(liste_elem.iloc[it]["pred"]) + ":" + str(liste_elem.iloc[it + 1]["pred"]) + "*" + str(liste_elem.iloc[it + 2]["pred"]))
                    else:
                        result += ("-" + str(liste_elem.iloc[it]["pred"]) + ":" + str(liste_elem.iloc[it + 1]["pred"]) + ":" + str(liste_elem.iloc[it + 2]["pred"]))
                    it += 2
                elif superposition(liste_elem.iloc[it],liste_elem.iloc[it+1],Q3_h):
                    result += ("-" + str(liste_elem.iloc[it]["pred"]) + ":" + str(liste_elem.iloc[it + 1]["pred"]))
                    it += 1
                else:
                    result += ("-" + str(liste_elem.iloc[it]["pred"]))
                it += 1
            return result

    
        encode = []
        for k in range(len(liste_colonne_elem)):
            encode.append(encodage(liste_colonne_elem[k]))
            
        manuel_de_codage = [""] * len(encode)
        st.header("Décodage complet ...")
        st.write("Possibilité de corriger directement les erreurs içi :")
        
        
        for k in range(len(encode)):
            manuel_de_codage[k] = st.text_area("Colonne " + str(k), value=encode[k], height=None, max_chars=None)
            
        sauver_texte = st.button("Sauver l'encodage dans un fichier texte")
        
        if sauver_texte:
            fichier = open('./data.csv', mode='a')
            for k in range(len(manuel_de_codage)):
                fichier.write(manuel_de_codage[k] + '\n.')
            fichier.close()  
            
    if option_choice == "Reconnaissance":
        message = st.write("Entrez un symbole")
        
        symbole = st.sidebar.selectbox(label="Liste des symboles", options=liste_h)
        
        st.sidebar.write("Le symbole " + symbole + " est le suivant !")
        img_symbole_select = st.sidebar.image(Image.open("./model/sample/elem_" + symbole +".png"))
        
        img_symbole = st.sidebar.file_uploader(label='Charger un symbole içi', type=['png', 'jpg'])
        
        if img_symbole:
#            message.empty()
            symbole = Image.open(img_symbole)
            st.sidebar.write("Symbole recherché")
            img = st.sidebar.image(symbole)
            
            
            symbole = np.array(symbole)
            img_data = cv2.resize(symbole, (100,100),interpolation=cv2.INTER_AREA)
            

            img_data = np.expand_dims(img_data,axis=-1)
            img_data = np.concatenate((img_data,img_data,img_data),axis= -1)
            img_data = np.expand_dims(img_data,axis=0)
            
            vecteur = model.predict(img_data)
            
            symbole_reconnus = liste_h[np.argmax(vecteur)]
            probabilite = vecteur[:,np.argmax(vecteur)]
            numero_du_symbole_reconnus = np.argmax(vecteur)

            st.sidebar.write("Ce symbole a était identifié comme le symbole : " + symbole_reconnus)
            
        else:
            numero_du_symbole_reconnus = liste_h.index(symbole)
            
        print(numero_du_symbole_reconnus)
        array = np.array(cropped_img)
        stock_cropped.empty()
            

        try:
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        except:
            gray = array
        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
              
        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
            
        element = []
        for i in range(0, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            Keep_area = area < 30000 and area > 10
            Keep_w_h = w <80 and h <80 and w > 3 and h> 3
                
 
            if all((Keep_area,Keep_w_h)):
                componentMask = (labels == i).astype("uint8") * 255
                componentMask = componentMask[y:(y+h),x:(x+w)]
                componentMask = cv2.bitwise_not(componentMask)
                    
                img_temp = cv2.resize(componentMask, (100,100),interpolation=cv2.INTER_AREA)
                    
                img_data = np.expand_dims(img_temp,axis=-1)
                img_data = np.concatenate((img_data,img_data,img_data),axis= -1)
                img_data = np.expand_dims(img_data,axis=0)
                
                vecteur = model.predict(img_data)
                    
                proba = vecteur[:,numero_du_symbole_reconnus]
                    
                element.append({"x":x,"y":y,"w":w,"h":h,"d":proba})
                    
            
        proba_max = st.slider("Les éléments possèdant une probabilité supérieure a :", min_value=0.00, max_value=1.00, value=0.5, step=0.01)
            
        for i in range(len(element)):
            if element[i]["d"] > proba_max:
                cv2.rectangle(array, (element[i]["x"], element[i]["y"]), (element[i]["x"] + element[i]["w"], element[i]["y"] + element[i]["h"]), col, 2)
                    
        stock_cropped.empty()
        array = cv2.resize(array, (1500,1000),interpolation=cv2.INTER_AREA)
        st.image(array, caption='')
        
    if option_choice == "Traduction":
        array = np.array(cropped_img)
        pass
    
    
    if option_choice == "Dessin":
        pass
    
    if save:
        cv2.imwrite("./results.png",array)
        st.write("Image enregistré dans le répertoire courant")
        st.sidebar.write("Image enregistré dans le répertoire courant")

elif option_choice == "Traduction":
    file = st.file_uploader("Importer un fichier texte au format manuel de codage",["csv","txt","png"])
    if file:
        df = pd.DataFrame(file)
        for i in range(df.shape[0]):
            annotated_text(str(df.iloc[i,].values))

        annotated_text(
        "This ",
        ("is", "verb", "#8ef"),
        " some ",
        ("annotated", "adj", "#faa"),
        ("text", "noun", "#afa"),
        " for those of ",
        ("you", "pronoun", "#fea"),
        " who ",
        ("like", "verb", "#8ef"),
        " this sort of ",
        ("thing", "noun", "#afa"),
        "."
        )
    
elif option_choice == "Dessin":
    drawing_mode = st.sidebar.selectbox(
            "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
            )

    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    

            # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=150,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
        )

        # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects) 
    
    recherche = st.button("Recherche des symbole match")
    
    if recherche:
        symbole = np.array(Image.fromarray(canvas_result.image_data,'RGBA').convert('L'))
              
        
        img_data = cv2.resize(symbole, (100,100),interpolation=cv2.INTER_AREA)
        print(img_data.shape)
        img_data = np.expand_dims(img_data,axis=-1)
        img_data = np.concatenate((img_data,img_data,img_data),axis= -1)
        
        img_data = np.expand_dims(img_data,axis=0)        
        vecteur = model.predict(img_data)[0]
        
        indice_top3 = heapq.nlargest(3, range(len(vecteur)), vecteur.take)

        symbole_reconnus = np.array(liste_h)[indice_top3]
        probabilite = np.round(vecteur[indice_top3],decimals = 2)
        
                
        st.write("Ce symbole est reconnus comme : ")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.header(symbole_reconnus[0] + " avec une probabilité de " + str(probabilite[0]))
            name = "model/sample/elem_" + symbole_reconnus[0] + ".png"
            st.image(Image.open(name))

        with col2:
            st.header(symbole_reconnus[1]+ " avec une probabilité de " + str(probabilite[1]))
            name = "model/sample/elem_" + symbole_reconnus[1] + ".png"
            st.image(Image.open(name))

        with col3:
            st.header(symbole_reconnus[2] + " avec une probabilité de " + str(probabilite[2]))
            name = "model/sample/elem_" + symbole_reconnus[2] + ".png"
            st.image(Image.open(name))

else:
    st.write(messages[langue]["2"])

    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)

        img_array = np.array(img)

        st.image(img_array)
 