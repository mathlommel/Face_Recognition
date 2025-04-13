# Face_Recognition
Dans ce projet, l'objectif est de concevoir une interface de reconnaissance faciale. L'objectif est assez simple : permettre à une personne de charger une image, et 
lancer un recherche par reconnaissance faciale. Cet algorithme devra renvoyer l'image la plus semblable à l'image chargée par l'utilisateur, avec un score de similarité.

## Approche théorique
Pour réaliser cette tâche de reconnaissance faciale, nous allons utiliser le réseau neuronal Inception-ResNet-V1, pré-entraîné sur le jeu de données VGGFace2. Ce jeu de donnée contient un batch de 
500 000 célébrités, avec plusieurs centaines d'images pour chaque individu. Cette base de donnée contient notamment des photos assez diversifiées pour chaque célébrité : images récentes,
images de l'individu plus jeune, images de ses différents profils,..etc. L'utilisation de ce modèle pré-entraîné nous permet d'utiliser un réseau de neurones dont les  performances sont bien 
supérieures à ce que nous pourrions réaliser à notre échelle d'étudiants.

Notre interface de reconnaissance faciale se basera sur un jeu de données plus petit que celui qui a servi à l'entraînement du modèle : ["Labelled Faces in the Wild (LFW) Dataset"](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset). Cette base d'images
est composée de 500 célébrités différentes, pour lesquelles plusieurs images peuvent être disponibles.

Le processus de reconnaissance faciale est le suivant : 
- Tout d'abord, nous construisons une base de donnée composée des embeddings (issus du réseau neuronal) de toutes les images de notre petite base de données
- Ensuite, lorsque l'utilisateur importe une image, nous calculons également l'embedding de celle-ci
- Finalement, nous déterminons l'image de la base de donnée dont l'embedding est "le plus proche" de l'embedding de l'image importée

Le modèle ResNet ayant été entraîné sur un très grand nombre d'images, et proposant de bonnes performances, notre implémentation est basée sur l'hypothèse que l'embedding obtenu avec ce modèle
caractérise des traits importants du visage en entrée. En suivant cette hypothèse, deux visages similaires (qui ont des caractéristiques physiques proches, comme la taille du nez, la couleur des
yeux, la couleur des cheveux,...), auront des embeddings dans l'espace latent qui seront eux aussi proches. La similarité des vecteurs dans l'espace latent correspondra en pratique au cosinus de 
l'angle entre les vecteurs. Cette similarité multipliée par 100 nous donne une quantité assimilable à un pourcentage de similarité (si le cosinus est proche de 1, la similairté est presque de 100%, 
et plus le cosinus diminue, plus l'angle entre les vecteurs est grand, et donc plus la similarité diminue).

A l'aide du modèle, nous passons d'une comparaison d'images tridimensionnelles (dimensions de l'image + canaux RGB) à une comparaison de vecteurs de longueur 1000. Cette simplification
est donc très intéressante, et plutôt pertinente, puisque si nous devions comparer nous même les visages, nous nous baserions nous aussi sur certains traits physiques de ceux-ci.

Enfin, en amont de cet embedding, les images d'entrée seront pré-processées via le modèle MTCNN, qui permet de localiser la position du visage sur l'image. Ceci est nécessaire pour que la reconnaissance
faciale se base bel et bien sur le visage, et sur lui seul (et non l'environnement extérieur de l'image).

En ce qui concerne le résultat de la reconnaissance faciale, la similarité sera accompagnée d'un label ('MATCH', 'PARTIAL MATCH' et 'NO MATCH'). Ces labels sont associés à des seuils de similarités.
En pratique, les similarités supérieures à 75% sont bien souvent bonnes, mais il est possible d'obtenir une reconnaissance correcte avec une similarité inférieure à ce seuil. Le label PARTIAL MATCH a donc
été ajouté, pour les similarités intermédiaires (50 à 75%). Enfin, le label NO MATCH est affiché lorsque la similarité est inférieure à 50%.

Cet ajout est assez essentiel lorsque l'on crée une interface basée sur l'intelligence artificielle. En effet, même si les performances sont très bonnes, des erreurs sont toujours possibles. Ces 3 labels
permettent donc à l'utilisateur d'établir lui même la conclusion : 
- si la similarité est supérieure à 75% (MATCH), l'individu testé peut être considéré comme reconnu
- si la similarité est intermédiaire (PARTIAL MATCH), l'interface propose un visage similaire, mais avec une certitude moindre : c'est à l'utilisateur de trancher
- si la similarité est faible, alors l'interface propose le visage le plus proche, mais en indiquant à l'utilisateur que la similarité n'est pas suffisante pour considérer l'individu comme reconnu

Ces labels permettent donc de réduire l'impact des erreurs potentielles de notre implémentation.

## Quelques Résultats
En pratique, les résultats sont plutôt bons. En effet, pour les individus présents dans le jeu de données LFW, la reconnaissance faciale est efficace, et renvoie bien l'individu souhaité.
De plus, nous avons aussi pu tester cette reconnaissance faciale avec des images qui n'étaient pas présentes dans la base de données. Comme nous l'avions espéré, notre code proposait alors 
un visage dont certains traits physiques étaient très similaires, ce qui démontre pratiquement l'intérêt de notre approche.

Cependant, nous avons aussi pu "piéger" notre algorithme. Pour tester les limites, nous avons tenté d'utiliser une image d'un individu qui est dans la base de données, mais dont la ressemblance 
n'est pas immédiate. Pour cela, nous avons trouvé une image de Jason Statham dans le film "Parker", où il est déguisé en prêtre, avec des lunettes et des cheveux gris. Cette image de cet acteur est
très différente de l'image présente dans le jeu de données, d'où l'intérêt de notre test. Comme nous pouvions le prévoir, le résultat était incorrect. L'individu le plus semblable à l'image importée
n'était pas Jason Statham, mais un autre acteur, qui porte des lunettes, a les cheveux gris, et la même forme du visage que Jason Statham.

Sur ce test simple, nous observons que notre approche présente certaines limites. Bien que l'embedding d'une image traduise bien les caractéristiques physique du visage, notre comparaison est établie
avec une base de donnée assez restreinte. Ainsi, pour un individu (comme Jason Statham), qui n'a qu'une seule image, et de qualité assez mauvaise, dans cette base, le changement de quelques 
caractéristiques physique peut complètement changer le résultat de notre algorithme.

Ceci nous montre que des améliorations sont possibles. Considérer une base de donnée plus grande pourrait être une solution. Nous pourrions alors paralléliser le calcul des similarité pour que cet ajout
n'influence pas le temps de calcul.

## Contributors
Salma ES-SAJRADI, Yoann FOURNIER, Mathias LOMMEL, Julie THOMAS

---------------------------------------
Machine Learning & Deep Learning

Département __Mathématiques Appliquées__

INSA Rennes, 2024-2025
