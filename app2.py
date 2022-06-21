########################################################################## 
################################################ Application OCR-PT-CT
##########################################################################

### Importation librairie ------
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
import string

### Loading model and label from repertory ------
def importation_model_et_label(repertoire):
    model = tf.keras.models.load_model(repertoire + '/model.hd5')
    labels = []
    with open(repertoire + '/labels', 'rb') as temp:
        labels = pickle.load(temp)
    labels = list(labels.values())
    return model,labels

model,liste_h = importation_model_et_label("./model")

# List of the symbole which the correlation sign must switch
liste_of_symbole_to_switch_correlation = ["A40","A41","A42","A43","A44","A45","A46","A47","A48","A49","A50","A51","B1","B2","B3","B4","B5","B6","B7",
                                          "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","F4"]




### Loading json file contaning the languages
fichier_json = open('./language.json', 'r', encoding="utf-8")

with fichier_json as fichier:
   messages = json.load(fichier)
   

### Set default language
langue = "EN"


##### Graphical part

### PAge configuration.
st.set_page_config(page_title="App", page_icon="⚕️", layout="centered", initial_sidebar_state="expanded")

# Place the logo on the sidebar.
logo = st.sidebar.image(cv2.resize(np.array(Image.open("images/logo.png")),(80,100)))

# Again the logo 
html_temp = """ 
    <img src="images/logo.png" alt = "LOGO"> 
    """
st.markdown(html_temp, unsafe_allow_html = True) 

# Choice for the language in the sidebar
langue = str(st.sidebar.selectbox(messages[langue]["language_choice"], options=["EN","FR","ESP"]))

# Title of app
html_temp = """ 
    <div style ="background-color:#af81c2;padding:13px"> 
    <h1 style ="color:black;text-align:center;">OCR-PT-CT</h1> 
    </div> 
    """
      
st.markdown(html_temp, unsafe_allow_html = True) 

# File uploader set to not show the encoding
st.set_option('deprecation.showfileUploaderEncoding', False)

# Title of the sidebar
title_alignment ="""
    <div style = "background-color:#b897c6">
    <h1 style = "color:blac;text-align:center;">Option de paramétrage</h1>
    </div>
    """
st.sidebar.markdown(title_alignment, unsafe_allow_html=True)

# File browser
img_file = st.sidebar.file_uploader(label=messages[langue]["1"], type=['png', 'jpg'])
# Box color for the rectangle
box_color = st.sidebar.color_picker(label=messages[langue]["countours"], value='#0000FF')
# Different choice of aspect ratio for the croped image
aspect_choice = st.sidebar.selectbox(label=messages[langue]["ratio"], options=["1:1", "16:9", "4:3", "2:3", "Libre"])
aspect_dict = {"1:1": (1, 1),"16:9": (16, 9),"4:3": (4, 3),"2:3": (2, 3),"Libre": None}
aspect_ratio = aspect_dict[aspect_choice]

# Choice between the option (Recognition, ...)
option_choice = st.sidebar.radio(label="Options", options=[messages[langue]["option1"],
                                                           messages[langue]["option2"],
                                                           messages[langue]["option3"],
                                                           messages[langue]["option4"],
                                                           messages[langue]["option5"]])

# Button to save the result put at the end of the sidebar
save = st.sidebar.button('Sauver le résultat')

### Change hexadecimale to RGB
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

### Get the correlation of an image
def correlation(img):
    x = []
    y = []
    n_x = 0
    n_y = 0
    for i in range(0,img.shape[0],1):
        n_x += 1
        n_y = 0
        for j in range(0,img.shape[1],1):
            n_y += 1
            if(img[i,j] == 255):
                x.append(n_x)
                y.append(n_y)
    y = list(reversed(y))
    res = np.corrcoef(x, y)[0,1]
    return res


# If a file is browsed
if img_file:
    col = hex_to_rgb(box_color)
    img = Image.open(img_file)
    # Cut the image with st_cropper
    cropped_img = st_cropper(img, realtime_update=True, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    # Getting and display the cropped image
    _ = cropped_img.thumbnail((1500,1500))
    stock_cropped = st.image(cropped_img)
    
    ### First option
    if option_choice == messages[langue]["option1"]:
        ### Get the image as an array
        array = np.array(cropped_img)
        ### Undisplay the cropped image
        stock_cropped.empty()
        
        ### If not already gray, convert to gray scale
        try:
            gray = cv2.cvtColor(array.copy(), cv2.COLOR_BGR2GRAY)
        except:
            gray = array.copy()
        
        # Looking for all the element in the image
        thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (numLabels, _, stats, centroids) = output
        
        # Loop over the different element 
        for i in range(0, numLabels):
            # Get the statistics of the element i
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            
            # Look if the element is big enougth and small enougth to be asume as a hyeroglyf
            Keep_area = area < 30000 and area > 10
            Keep_w_h = w <80 and h <80 and w > 2 and h> 2
            
            # If this is the case, draw a rectangle around it
            if all((Keep_area,Keep_w_h)):
                cv2.rectangle(array, (x, y), (x + w, y + h), col, 1)
        
        # Resize and display the new array
        array = cv2.resize(array, (500,1000),interpolation=cv2.INTER_AREA)
        st.image(array, caption='')
    
    ### Second option (Decode)
    if option_choice == messages[langue]["option2"]:
        
        # Get the cropped image as an array
        array = np.array(cropped_img)
        img = array.copy()
        stock_cropped.empty()
        # Convert to gray scale
        try:
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        except:
            gray = array
        
        # Look for all the connected component
        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]      
        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (numLabels, _, stats, centroids) = output
        
        # Init of a data frame how as the number of lines of the numbers of element
        frame = pd.DataFrame(columns = ["x"],index = list(range(numLabels)))
        
        # Loop over the element
        for i in range(0, numLabels):
            # Get individual statistic
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            
            # Deciding which one to keep
            Keep_area = area < 30000 and area > 10
            Keep_w_h = w <80 and h <80 and w > 3 and h> 3
            
            # If we keep it, stock the coordinate of is center on the absices axes in the frame
            if all((Keep_area,Keep_w_h)):
                frame.iloc[i,] = cX
                
        # Looking to eliminate the ones with no value, how are the one which there area, higth or width were suspect
        index_with_nan = frame.index[frame.isnull().any(axis=1)]
        frame.drop(index_with_nan,0, inplace=True)
        # Procede the same traitement on the centroid liste to keep only the one we want
        centroids = pd.DataFrame(centroids,columns = ["x","y"])
        centroids.drop(index_with_nan,0, inplace=True)

        ### Clustering the columns
        frame = np.array(frame)
        db = DBSCAN(eps=15, min_samples=4).fit(frame)
        labels = db.labels_

        # Get everything in to a new df.
        groupe = pd.DataFrame(np.array((labels,centroids["x"],centroids["y"])).transpose(),columns = ["groupe","x","y"])
        
        # Group by groupe (label)
        resultat = groupe.groupby(by='groupe', as_index=False).agg({'x': pd.Series.nunique})

        # The element labelised as -1 have no groups so we must drop them.
        idx = []
        for i in range(groupe.shape[0]):
            if groupe["groupe"][i]  == -1:
                idx.append(i)
        groupe = groupe.drop(idx)
        
        # Getting the statistic over all different group
        points = groupe.groupby("groupe").agg({"x":"mean","y":["max","min"]})
        moyenne_x_grp = points["x"]["mean"]

        # Sort the groups by there mean of x, the we get the order from left to rigth
        sort_moyenne_x_grp = np.sort(moyenne_x_grp)

        # Setting a distance which for the value is in the group. The distance is from the centroid of
        # the element and the mean of a group.
        # This is alowing us to assing a group to the element how are not part of a group
        ecart_a_x = 30
        (numLabels, _, stats, centroids) = output

        # elemement is going to be a liste of dictionnaries, were each dictionnaries contans info abput the symbol
        elements = []
        # Loop over the element
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
                if np.min(np.abs(sort_moyenne_x_grp - x)) < ecart_a_x: # We check if a group if at less distance then ecart_a_x from this element
                    groupe = np.argmin(np.abs(sort_moyenne_x_grp - x)) # We assign a group to the element
                    
                    ### Now that we have all the group, we can predict for all the element
                    # Cutting the element 
                    temp = img[y:(y+h),x:(x+w)] 
                    # Resizing to use the model
                    temp = cv2.resize(temp, (100,100),interpolation=cv2.INTER_AREA)
                    img_data = np.expand_dims(temp, axis=0)
                    # Predict
                    vecteur = model.predict(img_data)
                    # Get the label from our global variable liste_h
                    res = liste_h[np.argmax(vecteur)]
                    # append information to the list element
                    if res[0] in string.ascii_uppercase[:11]:
                        cor = correlation(cv2.threshold(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY), 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1])
                        if res in liste_of_symbole_to_switch_correlation:
                            cor = cor * -1
                        elements.append({"area":area,"x":x,"y":y,"w":w,"h":h,"cX":cX,"cY":cY,"pred":res,"cor":cor,"groupe":groupe})
                    else:    
                        elements.append({"area":area,"x":x,"y":y,"w":w,"h":h,"cX":cX,"cY":cY,"pred":res,"cor":0,"groupe":groupe})
                        
                    # Write the prediction on the cropped image
                    cv2.putText(array,str(res),(x + w -10, y + h),cv2.FONT_HERSHEY_TRIPLEX , 0.5, col,thickness = 1)
                    
        # Display the image
        array = cv2.resize(array, (300,1000),interpolation=cv2.INTER_AREA)
        st.image(array, caption='')
        
        # Get a data frame for each group to allow us to encode them separately
        frame = pd.DataFrame(elements)
        groupe_by_frame = frame.groupby("groupe")
        liste_colonne_elem = []
        for i in range(sort_moyenne_x_grp.shape[0]):
            liste_colonne_elem.append(groupe_by_frame.get_group(i))
            print(liste_colonne_elem[i])
    
        # Fonction to detect juxtaposition
        def juxtaposition(elem1,elem2):
            if elem2["cY"] > elem1["y"] and elem2["cY"] < (elem1["y"] + elem1["h"]):
                return True
            return False


        # Fonction to detect superposition up to 3 symbol
        def superposition(elem1,elem2,max_h,elem3 = None):
            if not elem3 is None:
                h = abs(elem3["y"] - (elem1["y"] + elem1["h"]))
            else:
                h = abs(elem2["y"] - (elem1["y"] + elem1["h"]))

            if h < max_h:
                return True
            return False

        # Fonction going threw a date frame of element form the same columns
        def encodage(liste_elem):
            orientation = liste_elem["cor"].mean()
            result = ""
            it = 0
            Q3_h = liste_elem["h"].quantile(0.75)  
            while it < (liste_elem.shape[0] - 2):
                if juxtaposition(liste_elem.iloc[it],liste_elem.iloc[it + 1]):
                    if orientation < 0:
                        result += ( "-" + str(liste_elem.iloc[it+1]["pred"]) + "*" + str(liste_elem.iloc[it]["pred"]))
                    else:       
                        result += ( "-" + str(liste_elem.iloc[it]["pred"]) + "*" + str(liste_elem.iloc[it + 1]["pred"]))
                    it += 1
                elif juxtaposition(liste_elem.iloc[it+1],liste_elem.iloc[it]) and juxtaposition(liste_elem.iloc[it+1],liste_elem.iloc[it+2]):
                    if orientation < 0:
                        result += ( "-" + "(" +str(liste_elem.iloc[it]["pred"]) + ":" + str(liste_elem.iloc[it + 1]["pred"]) + ")*" + str(liste_elem.iloc[it+2]["pred"])) 
                    else:
                        result += ( "-" + "(" +str(liste_elem.iloc[it]["pred"]) + ":" + str(liste_elem.iloc[it + 2]["pred"]) + ")*" + str(liste_elem.iloc[it+1]["pred"])) 
                elif superposition(liste_elem.iloc[it],liste_elem.iloc[it+1],Q3_h,elem3 = liste_elem.iloc[it+2]):
                    if juxtaposition(liste_elem.iloc[it +1],liste_elem.iloc[it + 2]):
                        if orientation < 0:
                            result += ("-" + str(liste_elem.iloc[it]["pred"]) + ":" + str(liste_elem.iloc[it + 2]["pred"]) + "*" + str(liste_elem.iloc[it + 1]["pred"]))
                        else:
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

        # Encode every columns
        encode = []
        for k in range(len(liste_colonne_elem)):
            encode.append(encodage(liste_colonne_elem[k]))
        
        # Create a list for the results and the comment made on the app
        manuel_de_codage = [""] * len(encode)
        comment = [""] * len(encode)
        # Display text on the app
        st.header("Décodage complet ...")
        st.write("Possibilité de corriger directement les erreurs içi :")
        
        # Display the encode text for each columns, with a blank place to write some specificities
        for k in range(len(encode)):
            if liste_colonne_elem[k]["cor"].mean() > 0:
                com = "Droite"
            else:
                com = "Gauche"
            comment[k] = st.text_area("Comment for colomns " + str(k), value = com,height=0,max_chars=200)
            manuel_de_codage[k] = st.text_area("Colums " + str(k), value=encode[k], height=None, max_chars=None)
            
        # Button for saving the result and the adresse to save it.
        file_to_put_results = st.text_area("adresse of the file ", value = "./data.csv",height=20,max_chars=100)
        sauver_texte = st.button("Sauver l'encodage dans un fichier texte")
        
        # Open the csv file and write the code and comments
        if sauver_texte:
            fichier = open(file_to_put_results, mode='a')
            for k in range(len(manuel_de_codage)):
                fichier.write("Comments : " + comment[k] + "\n")
                fichier.write(manuel_de_codage[k] + '\n.')
            fichier.close()  
            
    ### Option 3 : Recognition
    if option_choice == messages[langue]["option4"]:
        message = st.write("Entrez un symbole")
        
        
        # sidebar selectbox we've all the hyeroglyf knowned by the model
        symbole = st.sidebar.selectbox(label="Liste des symboles", options=liste_h)
        
        # Display the hyeroglyf chosen from the samples we've loaded before.
        st.sidebar.write("Le symbole " + symbole + " est le suivant !")
        img_symbole_select = st.sidebar.image(Image.open("./model/sample/elem_" + symbole +".png"))
        
        # Otherwise browse a file contaning the symbol
        img_symbole = st.sidebar.file_uploader(label='Charger un symbole içi', type=['png', 'jpg'])
        
        # looking for the class of the browse image
        if img_symbole:
#            message.empty()
            symbole = Image.open(img_symbole)
            st.sidebar.write("Symbole recherché")
            img = st.sidebar.image(symbole)
            
            # Converting to array and rescaling
            symbole = np.array(symbole)
            img_data = cv2.resize(symbole, (100,100),interpolation=cv2.INTER_AREA)
            img_data = np.expand_dims(img_data,axis=-1)
            img_data = np.concatenate((img_data,img_data,img_data),axis= -1)
            img_data = np.expand_dims(img_data,axis=0)
            # Predict
            vecteur = model.predict(img_data)
            # Assigning to a class
            symbole_reconnus = liste_h[np.argmax(vecteur)]
            probabilite = vecteur[:,np.argmax(vecteur)]
            numero_du_symbole_reconnus = np.argmax(vecteur)
            # Display the results of the prediction
            st.sidebar.write("Ce symbole a était identifié comme le symbole : " + symbole_reconnus)
        
        # Else we the index of the symbol chosen in the list
        else:
            numero_du_symbole_reconnus = liste_h.index(symbole)

        # undisplay the cropped image
        array = np.array(cropped_img)
        stock_cropped.empty()
            
        # Converting to gray scale
        try:
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        except:
            gray = array
        
        # Connected component 
        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
            
        # Element will be a list of dictionnaries conrtening info over the symbols
        element = []
        # Loop over the element
        for i in range(0, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            Keep_area = area < 30000 and area > 10
            Keep_w_h = w <80 and h <80 and w > 3 and h> 3
                
 
            if all((Keep_area,Keep_w_h)):
                # Apply a mask to keep only the symbol
                componentMask = (labels == i).astype("uint8") * 255
                componentMask = componentMask[y:(y+h),x:(x+w)]
                componentMask = cv2.bitwise_not(componentMask)
                
                # Resizing to predict
                img_temp = cv2.resize(componentMask, (100,100),interpolation=cv2.INTER_AREA)    
                img_data = np.expand_dims(img_temp,axis=-1)
                img_data = np.concatenate((img_data,img_data,img_data),axis= -1)
                img_data = np.expand_dims(img_data,axis=0)
                
                # Predict 
                vecteur = model.predict(img_data)
                
                # Get the probability for the symbol we are looking for
                proba = vecteur[:,numero_du_symbole_reconnus]
                
                # Append to our list
                element.append({"x":x,"y":y,"w":w,"h":h,"d":proba})
                    
        # Slider how is deciding the minimum probabilities to the symbol to circle
        proba_min = st.slider("Les éléments possèdant une probabilité supérieure a :", min_value=0.00, max_value=1.00, value=0.5, step=0.01)
        
        # Loop over our listes of element 
        for i in range(len(element)):
            # Drawing rectangle over the ones how are higer then the min_proba
            if element[i]["d"] > proba_min:
                cv2.rectangle(array, (element[i]["x"], element[i]["y"]), (element[i]["x"] + element[i]["w"], element[i]["y"] + element[i]["h"]), col, 2)
        
        # Display the results
        stock_cropped.empty()
        array = cv2.resize(array, (500,1000),interpolation=cv2.INTER_AREA)
        st.image(array, caption='')
    
    # We are in the section were we are working we've an image so no traduction
    if option_choice == messages[langue]["option3"]:
        array = np.array(cropped_img)
        pass
    
    # if the button save is pressed, the cropped image is save
    if save:
        cv2.imwrite("./results.png",array)
        st.write("Image enregistré dans le répertoire courant")
        st.sidebar.write("Image enregistré dans le répertoire courant")

#### Option 3 : traduction
elif option_choice == messages[langue]["option3"]:
    file = st.file_uploader("Importer un fichier texte au format manuel de codage",["csv","txt","png"])
    if file:
        # Convert file to data frame 
        df = pd.DataFrame(file)
        for i in range(df.shape[0]):
            # Annotate each columns
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

#### Option 5 : Drawning
elif option_choice == messages[langue]["option5"]:
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

    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) 
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        st.dataframe(objects) 
    
    recherche = st.button("Recherche des symbole match")
    
    if recherche:
        symbole = np.array(Image.fromarray(canvas_result.image_data,'RGBA').convert('L'))
              
        
        img_data = cv2.resize(symbole, (100,100),interpolation=cv2.INTER_AREA)
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

#### Else, image took with the camera
else:
    st.write(messages[langue]["2"])

    # Get the picture
    img_file_buffer = st.camera_input("Take a picture")
    
    
    if img_file_buffer is not None:
        # Recup the image
        img = Image.open(img_file_buffer)
        img_array = np.array(img)
        
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        except:
            gray = img_array
            
            
        threshold = st.slider("Thresold", min_value=0, max_value=255, value=127, step=1)

        
        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.bitwise_not(thresh)
        temp_thresh = st.image(thresh)
        
        # Button to launch, once the thresold is ready
        launch = st.button("Lancer la recherche une fois le treshold réglé")
    
        if launch:
            temp_thresh.empty()
            output = cv2.connectedComponentsWithStats(thresh, 253, cv2.CV_32S)
            (numLabels, _, stats, centroids) = output
                
            for i in range(0, numLabels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
    
                Keep_area = area < 30000 and area > 10
                Keep_w_h = w <800 and h <800 and w > 10 and h> 10
                
                
                if all((Keep_area,Keep_w_h)):
                    temp = thresh[x:x+w,y:y+h]
                    
                    try :
                        temp = cv2.resize(temp, (100,100),interpolation=cv2.INTER_AREA)
                    except: 
                        next
                    
                    temp = np.expand_dims(temp,-1)
                    temp = np.concatenate((temp,temp,temp),axis = 2)
                    temp = np.expand_dims(temp,axis=0)
                    
                    if (sum(temp.shape) != 204):
                        next
                    else:    
                        vecteur = model.predict(temp)
                        pred = liste_h[np.argmax(vecteur)]
                        
                        
                        cv2.rectangle(img_array, (x,y), (x+w, y+h), (255,0,0), 2)
                        cv2.rectangle(thresh, (x,y), (x+w, y+h), (255,0,0), 2)
                        cv2.putText(img_array,str(pred),(x + w, y + h),cv2.FONT_HERSHEY_TRIPLEX , 0.5,(255,0,0),thickness = 1)
        
            st.image(thresh)
            st.image(img_array)
        
        
        
        
        