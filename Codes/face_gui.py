import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import time
import random

import torchvision.transforms.functional as F

from evaluate import *


class FaceRecognitionApp:
    def __init__(self, root, path_data, path_embeddings, path_labels):
        self.root = root
        self.root.title("Face Recognition App")
        self.path_embeddings =  path_embeddings
        self.path_labels = path_labels
        self.path_data = path_data

        # Image TARGET
        self.label_target = Label(self.root, text="Target Image")
        self.label_target.grid(row=0, column=0)

        self.canvas_target = Canvas(self.root, width=300, height=300)
        self.canvas_target.grid(row=1, column=0)

        self.button_select = Button(self.root, text="Select Target Image", command=self.select_image)
        self.button_select.grid(row=2, column=0)  
        
        # Image PROCESSED
        self.label_processed = Label(self.root, text="Preprocessed Image")
        self.label_processed.grid(row=3, column=0)

        self.canvas_processed = Canvas(self.root, width=300, height=300)
        self.canvas_processed.grid(row=4, column=0)

        self.button_preprocess = Button(self.root, text="Preprocess", command=self.preprocess_image)
        self.button_preprocess.grid(row=6, column=0)


        # Résultat du MATCH
        self.label_result = Label(self.root, text="Matched Image")
        self.label_result.grid(row=0, column=1)

        self.canvas_result = Canvas(self.root, width=500, height=500)
        self.canvas_result.grid(row=1, rowspan = 4, column=1)

    
        self.button_match = Button(self.root, text="Match", command=self.match_image)
        self.button_match.grid(row=6, column=1)

        self.similarity_label = Label(self.root, text="Similarity: Not Available")
        self.similarity_label.grid(row=5, column=1)

        # Device sur lequel réaliser le matching
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Importation des modèles
        resnet, mtcnn = init_models(device)
        self.resnet = resnet
        self.mtcnn = mtcnn

    def select_image(self):
        """
        Sélectionne l'image target
        """
        # Récupération du chemin d'accès
        file_path = filedialog.askopenfilename()

        if file_path:
            # Affichage du nom du fichier
            self.path_target = file_path
            self.label_target.config(text=f"Target Image: {file_path.split('/')[-1]}")
            # Redimensionnement de l'image
            img = Image.open(file_path).resize((300, 300))

            # Affichage de l'image
            img_tk = ImageTk.PhotoImage(img)
            self.canvas_target.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas_target.image = img_tk

    def preprocess_image(self):
        """
        Calcule l'embedding de l'image cible.
        """
        # Target's embedding 
        img, img_crop, _ = preprocess_embedding(self.path_target, self.resnet, self.mtcnn)

        if img_crop is not None:
            # Récupération de l'image cropped
            ## Convertion en PIL.Image
            img_crop_pil = Image.fromarray((img_crop.numpy() * 255).astype(np.uint8))
            # ## Redimensionnement
            img_crop_resized = img_crop_pil.resize((300, 300))

            # Affichage de l'image cropped
            img_crop_tk = ImageTk.PhotoImage(img_crop_resized)
            self.canvas_processed.create_image(0, 0, anchor=tk.NW, image=img_crop_tk)
            self.canvas_processed.image = img_crop_tk

    def match_image(self):
        """
        Détermination de l'image "match".
        """
        ############################
        ## DETERMINATION DU MATCH ##
        ############################
        
        # Si besoin, collecte des embeddings de la base de données
        ## (Si de nouvelles images sont ajoutées à la base de données, les fichiers .csv sont mis à jour)
        compute_embeddings(args.path_data, args.path_embeddings, args.path_labels, self.resnet, self.mtcnn)

        # Recherche du meilleur match parmi les images de la base de données
        img_target, img_crop_target, similarity, best_img, best_name, best_index = search(self.path_target, self.path_embeddings, self.path_labels, self.resnet, self.mtcnn)

        ############################
        #### AFFICHAGE DYNAMIQUE ####
        ############################
        # Récupération du tableau de labels
        labels = np.loadtxt(self.path_labels, delimiter=",", dtype=str)

        # Sélection d'environ 100 images au hasard (pour l'affichage)
        pics = [random.randint(0, len(labels)) for i in range(100)]
        # On retire les doublons
        index_pics = np.unique(pics)
        # On ajoute l'image matchée
        index_pics = np.append(index_pics, best_index)

        # Liste des labels que nous allons afficher à l'écran
        labels_show = labels[index_pics]

        # Calcul de la couleur d'encadrement (rouge ou vert)
        if similarity >= 75:
            border_color = "green"
            match_text = "MATCH"
            match_color = "green"
        elif similarity < 75 and similarity >= 50:
            border_color = "orange"
            match_text = "PARTIAL MATCH"
            match_color = "orange"
        else:
            border_color = "red"
            match_text = "NO MATCH"
            match_color = "red"

        # Simuler le défilement des images
        for i, label in enumerate(labels_show):
            img_path = label
            img = Image.open(img_path).resize((500, 500))
            img_tk = ImageTk.PhotoImage(img)

            # Encadrer l'image matchée en vert ou en rouge selon la similarité (pour la dernière image)
            if i == len(labels_show) - 1:  
                self.canvas_result.create_rectangle(
                    0, 0, 500, 500,
                    outline=border_color, width=10
                )

            # Afficher l'image
            self.canvas_result.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas_result.image = img_tk
            self.root.update()

            time.sleep(0.05)  # Pause pour simuler le défilement

        # Affichage du nom et de la similarité
        true_name = labels[best_index].split('/')[2]
        names = true_name.split('_')

        self.similarity_label.config(text="Similarity: "+str(np.round(similarity, 2))+" %      -      Name: "+str(names[0])+" "+str(names[1]))

        # Ajout du texte "MATCH"
        self.canvas_result.create_text(
            250, 450, text=match_text, font=("Helvetica", 30), fill=match_color, anchor="center"
        )


if __name__ == "__main__":
    # Récuparation des arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data",type=str, default="data/lfw")
    parser.add_argument("--path_embeddings",type=str,default="data/embeddings.csv")
    parser.add_argument("--path_labels",type=str,default="data/labels.csv")
    args = parser.parse_args()
    
    # Création de l'interface dynamique
    with torch.no_grad():

        root = tk.Tk()
        app = FaceRecognitionApp(root, args.path_data, args.path_embeddings, args.path_labels)
        root.mainloop()
