from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import numpy as np
import pandas as pd
import os
import sys
import cv2
from PIL import Image

import torchvision.transforms as T
import onnx
import onnxruntime

import matplotlib.pyplot as plt
import tqdm

import argparse 
import csv


def init_models(device):
    """
    Initialise les modèles utilisés pour l'embedding.

    Entrée : 
        - device : torch.cuda.device(device) - Device choisi

    Sorties :
        - resnet : InceptionResnetV1 - Modèle pour l'embedding
        - mtcnn  : MTCNN             - Modèle pour la localisation du visage
    
    """
    transform = T.ToPILImage()
    
    # Chargement des deux modèles
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(keep_all=True, device=device)

    return resnet, mtcnn


def preprocess_embedding(path,resnet,mtcnn):
    """
    Importation de l'image, de l'image cropped et de l'embedding associé.

    Entrée : 
        - path : String - Chemin d'accès à la photo

    Sorties : 
        - img       : Tensor - Image initiale
        - img_crop  : Tensor - Image cropped
        - embedding : Tensor - Embedding de l'image
    """

    # Importation de l'image
    img = Image.open(path)

    if img.mode == 'L':
        # Convertir l'image en RGB
        img = img.convert('RGB')

    # Preprocessing de l'image
    img_cropped = mtcnn(img)

    if img_cropped is not None:
        # Embedding de l'image
        embedding = resnet(img_cropped) 

        # Remise de l'image cropped dans la bonne dimension
        img_crop = torch.transpose(img_cropped[0],0,2)
        img_crop = torch.transpose(img_crop,0,1)

        return img, img_crop, embedding
    else:
        print("ERROR : No face found.")
        return img, None, None


def compute_embeddings(path_data,name_embedd,name_label,resnet,mtcnn):
    """
    Construit une table d'embedding pour toutes les images de la base de données.

    Entrées : 
        - path_data   : String - Chemin d'accès aux données
        - name_embedd : String - Nom du fichier de sauvegarde des embeddings
        - name_label  : String - Nom de fichier de sauvegarde des labels de chaque embedding

        - resnet : InceptionResnetV1 - Modèle pour l'embedding
        - mtcnn  : MTCNN             - Modèle pour la localisation du visage

    Sorties : 
        - np.array(mat_embeddings) : Array - Matrice des embeddings
        - labels                   : Array - Vecteur des labels de chaque embedding
    """

    with torch.no_grad():

        # Si des fichiers de sauvegarde existent déjà
        if os.path.exists(name_embedd) and os.path.exists(name_label):
            # Ouverture de la matrice d'embeddings
            mat_embeddings = np.loadtxt(name_embedd,delimiter=",").tolist()     
            # Ouverture des labels
            labels = np.loadtxt(name_label,delimiter=",",dtype=str)

            # Récupération des noms de dossier
            name_folders = os.listdir(path_data)

            # On parcourt les dossiers qui ne sont pas dans le .csv
            name_labels = np.array([path.split('/')[-2] for path in labels])
            name_folders = np.setdiff1d(name_folders, name_labels)  

        else:
            # Création des deux fichiers
            with open(name_embedd, mode='w', newline='') as file:
                writer = csv.writer(file)

            with open(name_label, mode='w', newline='') as file:
                writer = csv.writer(file)


            # Initialisation de la matrice d'embeddings
            mat_embeddings = []
            # Initialisation du vecteur de labels
            labels = np.array([])
            
            # Récupération des noms de fichier
            name_folders = os.listdir(path_data)

        #On parcourt tous les dossiers souhaités
        for i, name in enumerate(name_folders):
            # Chemin d'accès du dossier de la personalité
            name_folder = os.listdir(path_data+"/"+name)

            for picture in name_folder:
                # Chemin d'accès de l'image
                path_image = path_data+"/"+name+"/"+picture
                # Création de l'embedding de l'image
                img, img_crop, embedding = preprocess_embedding(path_image,resnet,mtcnn)
                
                # Si un visage a été détecté
                if img_crop is not None:
                    # Ajout de l'embedding et du label dans les tableaux
                    mat_embeddings.append(embedding[0])
                    labels = np.append(labels,path_image)

            # Sauvegarde des matrices tous les 50 dossiers
            if i%50 == 0:
                np.savetxt(name_embedd, mat_embeddings, delimiter=',')
                np.savetxt(name_label, labels, delimiter=',',fmt='%s')

            # Affichage de la barre d'état
            s = "■"
            nb = int(np.floor(50*i/(len(name_folders))))
            chaine = "["+ "".join([s] * nb) + "".join([" "] * (50-nb)) +"]"
            sys.stdout.flush()
            sys.stdout.write("\r" + chaine) 

        # Sauvegarde finale
        np.savetxt(name_embedd, mat_embeddings, delimiter=',')
        np.savetxt(name_label, labels, delimiter=',',fmt='%s')

        return np.array(mat_embeddings), labels
    

def search(path_target,path_embedd, path_labels, resnet, mtcnn):
    """
    Cherche le match optimal entre une image target et la base de données disponible.

    Entrées : 
        - path_target : String - Chemin d'accès de l'image cible
        - path_embedd : String - Chemin d'accès du fichier d'embeddings
        - path_labels : String - Chemin d'accès 

        - resnet : InceptionResnetV1 - Modèle pour l'embedding
        - mtcnn  : MTCNN             - Modèle pour la localisation du visage

    Sorties :
        - img_target      : Tensor - Image cible
        - img_crop_target : Tensor - Visage cible
        - similarity      : Float  - % de similarité entre la cible et le match
        - best_img        : Array  - Image du meilleur match
        - best_name       : String - Nom du meilleur match
    """

    # Calcul de l'embedding de l'image cible
    img_target, img_crop_target, embedding_target = preprocess_embedding(path_target,resnet,mtcnn)

    if embedding_target is not None:
        embedding_target = embedding_target.detach().numpy()

        # Importation des embeddings de la base de données
        mat_embeddings = np.loadtxt(path_embedd,delimiter=",")
        labels = np.loadtxt(path_labels,delimiter=",",dtype=str)

        # Calcul de la norme 2 entre la cible et toute la base
        dist = map(lambda x: np.linalg.norm(x-embedding_target), mat_embeddings)
        dist_list = list(dist)
        
        # Récupération du chemin d'accès au meilleur match
        best_index = np.argmin(dist_list)
        label_img = labels[best_index]

        # Calcul du meilleur taux de similarité (normalisé entre 0 et 1)
        similarity = np.min(dist_list)/np.max(dist_list)
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = 100*cos(torch.from_numpy(embedding_target), torch.from_numpy(mat_embeddings[best_index,])).numpy()[0]
    
        # Importation de l'image via le label
        best_img = Image.open(label_img)
        
        # Nom du meilleur match
        best_name = label_img.split("/")[-2]
        
        return img_target, img_crop_target, similarity, best_img, best_name, best_index
    
    else:
        print("ERROR : No face found on the target image.")
        return None, None,None,None,None
            

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data",type=str, default="data/lfw")
    parser.add_argument("--path_embeddings",type=str,default="data/embeddings.csv")
    parser.add_argument("--path_labels",type=str,default="data/labels.csv")
    parser.add_argument("--path_target",type=str,default="data/own_data/Aaron_Peirsol.jpg")
    args = parser.parse_args()
    
    with torch.no_grad():
        # Device sur lequel réaliser le matching
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Importation des modèles
        resnet, mtcnn = init_models(device)
        
        # Si besoin, collecte des embeddings de la base de données
        compute_embeddings(args.path_data,args.path_embeddings,args.path_labels, resnet, mtcnn)
        
        # Recherche du meilleur match parmi les images de la base de données
        img_target, img_crop_target, similarity, best_img, best_name, best_index  = search(args.path_target,args.path_embeddings,args.path_labels, resnet, mtcnn)

        if img_target is not None:
            # Affichage et sauvegarde de la figure
            _,axs = plt.subplots(2,2,figsize=(10,10))
            axs[0,0].imshow(img_target)
            axs[0,0].set_title("Original")
            axs[1,0].imshow(img_crop_target)
            axs[1,0].set_title("Original cropped")

            axs[0,1].imshow(best_img)
            axs[0,1].set_title(f"Match - {best_name} - {np.round(similarity,3)}")

            # Ajustement du layout pour éviter le chevauchement
            plt.tight_layout()

            # Sauvegarde de la figure
            plt.savefig('Image_search.png')
            plt.show()
