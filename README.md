# TP3 : Support Vector Machines (SVM)

## Description
Ce TP a pour objectif d'apprendre à utiliser les SVM pour classer des données à l'aide du package `sci-kit learn` en Python et de la base de données `iris`. Nous avons utilisé, pour ce faire, différents noyaux (linéaire et polynomial) afin d'obtenir différents types de séparation. Nous avons également utilisé la méthode de validation croisée pour évaluer la performance de nos modèles. Nous avons mis en évidence l'influence du paramètre de régularisation C sur la performance de nos différents modèles. 

Dans le but de compredre un peu mieux les SVM et l'importance du paramètre C sur la séparation de nos données par le classifieur, nous avons utilisé une interface disponible en compilant le code présent dans le fichier `svm_gui.py` qui se situe dans le dossier `src`.

Pour finir, nous avons utilisé une base de données contenant des visages disponible [ici](http://vis-www.cs.umass.edu/lfw/) afin de faire de la reconnaissance faciale à l'aide de nos modèles, d'étudier l'impact de l'ajout de variables de bruit et d'observer l'amélioration des performances lorsqu'on réduit les dimensions de nos données bruitées à l'aide de l'ACP.

## Organisation
Le dossier `src` contient les fichiers suivants :

- `svm_script.py` : le script principal du TP
- `svm_source.py` : le code source fourni
- `svm_gui.py` : le code pour l'interface graphique

Le dossier `report` contient les fichiers suivants:

- `TP3_BOULAND.tex` : le fichier .tex du rapport
- `TP3_BOULAND.pdf` : le rapport au format pdf

Le dossier `outputs` contient les images nécessaires à la compilation du rapport.

Ce projet est sous licence MIT. Pour plus d'informations, veuillez consulter le fichier `LICENSE`.

## Auteur
Guillaume BOULAND ([https://github.com/guibouland](https://github.com/guibouland))

## Prérequis
Pour exécuter l'ensemble du code présent dans ce projet, il est nécessaire d'avoir une version de Python récente et d'installer les packages suivants: 

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `pylab`

De plus, il est nécessaire d'avoir un compilateur LateX pour générer le rapport.

## Installation
Pour cloner le dépôt, exécutez la commande suivante:
```bash
git clone git@github.com:guibouland/TP3_AP_SVM.git
```


