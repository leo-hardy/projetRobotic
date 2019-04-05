"# projetRobotic" 

Créer un algo de machine learning (réseau de neurones) pour que le robot pioneer se déplace jusqu'à la cible.
Simulation sur V-rep

L'entraînement du réseau se fait via une base d'exemples (points de l'espace discrétisés) à partir desquels on définit des commandes élémentaires.

La base du code pour la commande du robot vient d'un autre projet.



FAIT : construction de la liste des points et des commandes correspondantes et d'un fichier Offline_trainer.py

A FAIRE : 
 - apprentissage : écrire la méthode train() (dans Offline_trainer) , calcul de l'erreur ,  rétroprog avec propag des erreurs
 - run : calcul position relative de la cible, vérifier qu'on envoie la commande aux roues
 - ne pas oublier de faire * M pour la vitesse aux roues (voir normalisation par L et pi)
 
 méthode à suivre : pour entrainer le réseau faire une seule boucle où passent tous les exemples, puis ensuite calculer une erreur moyenne pour le gradient


Remarques : slide 147 erreur pour les commandes 2, 3, 5 et 6


PROBLEMES :
 - 1e essai : On avait systématiquement 1 seul parcours de la base d'apprentissage. La direction "droite" était priviligée légèrement.
 on l'a forcé à faire 100 parcours de la boucle
