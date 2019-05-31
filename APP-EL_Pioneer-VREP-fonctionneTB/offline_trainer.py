# coding: utf-8

import time
import math
import matplotlib.pyplot as pl

L = 3  # demi-longueur du carre explorable
i, j, k = 1, 1, 1  # coefficients de discretisations
theta = 0
M = 1  # vitesse angulaire max des roues en sortie
position = [0, 0, 0]
pi = math.pi
sample_position = []  # liste des positions cibles (ds le repere du robot) finales, [x,y,orientation_cible]
sample_command = []  # liste des commandes des roues finales

# generation de la base de test normalisee
# boucle pour generer les combinaisons 1 a 6
for i in [1 / 4, 1 / 2, 3 / 4, 1]:
    for j in [1 / 4, 1 / 2, 3 / 4, 1]:
        for theta in [-pi, -(3 * pi) / 4, -pi / 2, -pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4]:
            sample_position += [[1 * i, 0, theta / pi],  # 1
                                [1 * i, 1 * j, theta / pi],  # 2
                                [-1 * i, 1 * j, theta / pi],  # 3
                                [-1 * i, 0, theta / pi],  # 4
                                [-1 * i, -1 * j, theta / pi],  # 5
                                [1 * i, -1 * j, theta / pi]]  # 6

            sample_command += [[1, 1],  # 1
                               [-1, 1],  # 2
                               [1, -1],  # 3
                               [-1, -1],  # 4
                               [-1, 1],  # 5
                               [1, -1]]  # 6

# boucle pour generer les combinaisons 7 et 8
for k in [(3 / 4), (1 / 2), (1 / 4)]:
    sample_position += [[0, 0, -1 * (pi * k) / (pi)],  # 7
                        [0, 0, pi * k / (pi)]]  # 8

    sample_command += [[1, -1],  # 7
                       [-1, 1]]  # 8

# generation du point final   #9
sample_position += [[0, 0, 0]]  # len(sample_position) = 775
sample_command += [[0, 0]]

assert(len(sample_position) == len(sample_command))

#Generation de la base de test
test_position = []
test_command = []

for i in [1/3, 2/3]:
    for j in [1/3, 2/3]:
        for theta in [-(7 * pi) / 8, -(5 * pi) / 8, -(3 * pi) / 8, -(1 * pi) / 8, (1 * pi) / 8, (3 * pi) / 8, (5 * pi) / 8, (7 * pi) / 8]:
            test_position += [[1 * i, 0, theta / pi],  # 1
                                [1 * i, 1 * j, theta / pi],  # 2
                                [-1 * i, 1 * j, theta / pi],  # 3
                                [-1 * i, 0, theta / pi],  # 4
                                [-1 * i, -1 * j, theta / pi],  # 5
                                [1 * i, -1 * j, theta / pi]]  # 6

            test_command += [[1, 1],  # 1
                            [-1, 1],  # 2
                            [1, -1],  # 3
                            [-1, -1],  # 4
                            [-1, 1],  # 5
                            [1, -1]]  # 6
            
for k in [(7 / 8), (5 / 8), (3 / 8), (1/8)]:
    test_position += [[0, 0, -1*k],  # 7
                    [0, 0, k]]  # 8
            
    test_command += [[1, -1],  # 7
                    [-1, 1]]  # 8

assert(len(test_position))==len(test_command)
#len(test_position) = 200
#print('la base de test à une taille de :' + str(len(test_position)))     

# liste des positions tests
# L_pos = [[L*i,0,0],#1
# [L*i,L*j,0],#2
# [-L*i,L*j,0],#3
# [-L*i,0,0],#4
# [-L*i,-L*j,0],#5
# [L*i,-L*j,0],#6
# [0,0,-1*(pi*k)],#7
# [0,0,pi*k],#8
# [0,0,0]]#9

# liste des sorties desirees
# L_q = [[M,M],#1
# [M,-M],#2
# [-M,M],#3
# [-M,-M],#4
# [M,-M],#5
# [-M,M],#6
# [M,-M],#7
# [-M,M],#8
# [0,0]]#9

def theta_s(x, y):
    if x > 0:
        return 1 * math.atan(1 * y)
    if x <= 0:
        return 1 * math.atan(-1 * y)


class OfflineTrainer:
    def __init__(self, robot, NN):
        """
        Args:
            robot (Robot): a robot instance following the pattern of
                VrepPioneerSimulation
            target (list): the target position [x,y,theta]
        """
        self.robot = robot
        self.network = NN

        self.alpha = [1 / (2 * L), 1 / (2 * L),
                      1 / (math.pi)]  # normalition avec limite du monde cartesien = -3m � + 3m

    # pourquoi pas avec 1/L ? sinon on va de -0.5 a 0.5, de même avec 1/2pi !

    # il faut que quand training soit True l'apprentissage soit lance
    # d'un bloc. Il faut qu'il soit False qd le robot bouge
    
    
    def train(self, target):
        
        n_it = 0

        somme_erreur_av = [len(sample_position) * 4, len(sample_position) * 4]  # erreur maximum,
        #print("\nsomme_erreure maximale (calcul initial) = " + str(somme_erreur_av)+'\n')
        

        while self.training:

            L_erreures_normalisees_base_training = []
            L_erreures_normalisees_base_test = []
            
            #print(sample_position)
            for k in range(20):
                
                # calcul de l'erreur cummulée sur toute la base d'exemple
                
                somme_erreur = [0, 0]  #pour le gradient
                somme_erreur_carré = [0,0]
                somme_erreur_carré_test = [0,0]
                              
                
                for i in range(len(sample_position)):
                    command = self.network.runNN(sample_position[i])  # propage erreur et calcule la  vitesse des roues instant t
                    erreur = [(command[0] - sample_command[i][0]), (command[1] - sample_command[i][1])]
                    somme_erreur[0] += erreur[0]
                    somme_erreur[1] += erreur[1]
                    
                    somme_erreur_carré[0] += erreur[0]**2
                    somme_erreur_carré[1] += erreur[1]**2
                    
                    #self.network.backPropagate(erreur, 0.0001, 0) ne marche pas bien
                    
                    '''
                    if (i == 10) :
                        print("Pour le "+str(i)+"ème cas de la base d'exemple (" +str(sample_position[i])+")")
                        print("commande pour la roue gauche reçue : " + str(command[0]))
                        print("commande pour la roue gauche attendue : " + str(sample_command[i][0])+"\n")  
                    '''
                    
                #ajout de l'erreure au carré normaliséé par la taille de la base d'apprentissage et moyenné sur les 2 roues
                L_erreures_normalisees_base_training.append((somme_erreur_carré[0]/(1 * len(sample_position)) + somme_erreur_carré[1]/(1 * len(sample_position)))                 / 2 )

                #print("A l'itération " + str(n_it) + ", somme_erreur_carré = "+str(somme_erreur_carré))

                grad = [0, 0]
                grad[0] = somme_erreur[0] / (1 * len(sample_position))  # erreur moyenne
                grad[1] = somme_erreur[1] / (1 * len(sample_position))

                self.network.backPropagate(grad, 0.01, 0)  # grad, pas d'app, moment : permet de lisser la trajectoire
                somme_erreur_av = somme_erreur
                n_it += 1
                #Fin de l'itération pour la base d'apprentissage, début pour la base de test
                for i in range(len(test_position)):
                    
                    command = self.network.runNN(test_position[i])  # propage erreur et calcule la  vitesse des roues instant t
                    erreur_test = [(command[0] - test_command[i][0]), (command[1] - test_command[i][1])]
                    somme_erreur_carré_test[0] += erreur_test[0]**2
                    somme_erreur_carré_test[1] += erreur_test[1]**2
                    
                L_erreures_normalisees_base_test.append((somme_erreur_carré_test[0]/(1 * len(test_position)) + somme_erreur_carré_test[1]/(1 * len(test_position))) / 2 )




            #Tracé des courbes
            
            pl.clf()
            X = [i+1 for i in range(len(L_erreures_normalisees_base_training))]
            X2 = [i+1 for i in range(len(L_erreures_normalisees_base_training))]
            #print('\nX = ' + str(X))
            #print('\nL_erreures_normalisees_base_training = ' + str(L_erreures_normalisees_base_training))
            #print('\nL_erreures_normalisees_base_test = ' + str(L_erreures_normalisees_base_test))
            
            pl.plot(X,L_erreures_normalisees_base_training,'r+')
            pl.plot(X,L_erreures_normalisees_base_test,'bo')
            
            #base d'entrainement affichée avec des croix rouges
            #base de test affichée avec des ronds bleu
            
            pl.show()
            
            self.training = False

            #print("\nsomme_erreur finale = [" + str(somme_erreur[0]) + "," + str(somme_erreur[1]) + "]")
            print("\nTraining done after " + str(n_it) + " iterations !")

            # version avec arret quand l'erreure augmente
            # if (somme_erreur[0]+somme_erreur[1]) < (somme_erreur_av[0]+somme_erreur_av[1]) :
            # self.network.backPropagate(grad, 0.9,0) # grad, pas d'app, moment : permet de lisser la trajectoire
            # somme_erreur_av = somme_erreur
            # n_it+=1
            ##print("n_it = "+ str(n_it)+"\n")

            # else :
            # self.training = False
            # print("Training done after " + str(n_it) +" iterations !")

        while self.running:
            debut = time.time()

            position = self.robot.get_position()

            network_input = [0, 0, 0]

            # calcul de la position relative de la cible dans le referentiel du robot
            network_input[0] = ((target[0] - position[0]) * math.cos(position[2]) + (target[1] - position[1]) * math.sin(position[2])) * self.alpha[0]
            network_input[1] = ((target[0] - position[0]) * (-1)*math.sin(position[2]) + (target[1] - position[1]) * math.cos(position[2])) * self.alpha[1]
            network_input[2] = (-1)*(position[2] - target[2]) * self.alpha[2]

            command = self.network.runNN(network_input)  # propage erreur et calcul vitesses roues instant t
            print("command =" + str(command))


            #faut il mmultiplier la commande par certains paramètres pour "dénormaliser la commande" ?
            self.robot.set_motor_velocity(command)  # applique vitesses roues instant t,
            time.sleep(0.050)  # attend delta t

        self.robot.set_motor_velocity([0, 0])  # stop  apres arret  du prog d'app
        # position = self.robot.get_position() #  obtient nvlle pos robot instant t+1
        # Teta_t=position[2]

        self.running = False
