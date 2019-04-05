import time
import math

L = 3 #demi-longueur du carré explorable
i,j,k = 1,1,1 #coefficients de discrétisations 
theta = 0
M = 1 #vitesse angulaire max des roues en sortie
position= [0,0,0]
pi = math.pi
sample_position = [] #liste des positions cibles (ds le repère du robot) finales
sample_command = [] #liste des commandes des roues finales

#génération de la base de test
#boucle pour générer les combinaisons 1 à 6
for i in [1/4, 1/2, 3/4, 1] :
    for j in [1/4, 1/2, 3/4, 1] :
        for theta in [-pi ,-(3*pi)/4,-pi/2,-pi/4,0,pi/4,pi/2,3*pi/4] :
            sample_position += [[L/i,0,theta],#1
            [L*i,L*j,theta],#2
            [-L*i,L*j,theta],#3
            [-L*i,0,theta],#4
            [-L*i,-L*j,theta],#5
            [L*i,-L*j,theta]]#6
            
            sample_command += [[M,M],#1
            [M,-M],#2
            [-M,M],#3
            [-M,-M],#4
            [M,-M],#5
            [-M,M]]#6
            
#boucle pour générer les combinaisons 7 et 8
for k in [(3/4),(1/2),(1/4)]:
    sample_position += [[0,0,-1*(pi*k)],#7 
    [0,0,pi*k]]#8
    
    sample_command += [[M,-M],#7
    [-M,M]]#8

#génération du point final   #9
sample_position += [[0,0,0]]
sample_command += [[0,0]]
    
                

#liste des positions tests
#L_pos = [[L*i,0,0],#1
         #[L*i,L*j,0],#2
         #[-L*i,L*j,0],#3
         #[-L*i,0,0],#4
         #[-L*i,-L*j,0],#5
         #[L*i,-L*j,0],#6
         #[0,0,-1*(pi*k)],#7
         #[0,0,pi*k],#8
         #[0,0,0]]#9

#liste des sorties désirées
#L_q = [[M,M],#1
       #[M,-M],#2
       #[-M,M],#3
       #[-M,-M],#4
       #[M,-M],#5
       #[-M,M],#6
       #[M,-M],#7
       #[-M,M],#8
       #[0,0]]#9
       
def theta_s(x,y):
    if x>0:
        return 1*math.atan(1*y)
    if x<=0:
        
        return 1*math.atan(-1*y)
    



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

        self.alpha = [1/(2*L),1/(2*L),1/(math.pi)]  # normalition avec limite du monde cartesien = -3m à + 3m
        #pourquoi pas avec 1/L ? sinon on va de -0.5 à 0.5.

	#il faut que quand training soit True l'apprentissage soit lancé d'un bloc. Il faut qu'il soit False qd le robot bouge
    def train(self, target):
        











		position = self.robot.get_position()

        network_input = [0, 0, 0]
        network_input[0] = (position[0]-target[0])*self.alpha[0]
        network_input[1] = (position[1]-target[1])*self.alpha[1]
        network_input[2] = (position[2]-target[2]-theta_s(position[0], position[1]))*self.alpha[2]
        #Teta_t = 0

        while self.running:
            debut = time.time()
            command = self.network.runNN(network_input) # propage erreur et calcul vitesses roues instant t
            
            #pondération des erreurs           
            alpha_x = 1/(2*L)
            alpha_y = 1/(2*L)
            alpha_teta = 1.0/(math.pi)
            
            #fonction de coût avant le déplacement            
            crit_av= alpha_x*alpha_x*(position[0]-target[0])*(position[0]-target[0]) + alpha_y*alpha_y*(position[1]-target[1])*(position[1]-target[1]) + alpha_teta*alpha_teta*(position[2]-target[2]-theta_s(position[0], position[1]))*(position[2]-target[2]-theta_s(position[0], position[1]))  
            
                       
            self.robot.set_motor_velocity(command) # applique vitesses roues instant t,                     
            time.sleep(0.050) # attend delta t
            position = self.robot.get_position() #  obtient nvlle pos robot instant t+1       
            
            network_input[0] = (position[0]-target[0])*self.alpha[0]
            network_input[1] = (position[1]-target[1])*self.alpha[1]
            network_input[2] = (position[2]-target[2]-theta_s(position[0], position[1]))*self.alpha[2]
            
            #fonction de coût après le déplacement 
            crit_ap= alpha_x*alpha_x*(position[0]-target[0])*(position[0]-target[0]) + alpha_y*alpha_y*(position[1]-target[1])*(position[1]-target[1]) + alpha_teta*alpha_teta*(position[2]-target[2]-theta_s(position[0], position[1]))*(position[2]-target[2]-theta_s(position[0], position[1])) 

            if self.training:
                delta_t = (time.time()-debut)

                #moyenne des erreurs 
                grad = [
                    (-2/delta_t)*(alpha_x*alpha_x*(position[0]-target[0])*delta_t*self.robot.r*math.cos(position[2])
                    +alpha_y*alpha_y*(position[1]-target[1])*delta_t*self.robot.r*math.sin(position[2])
                    -alpha_teta*alpha_teta*(position[2]-target[2]-theta_s(position[0], position[1]))*delta_t*self.robot.r/(2*self.robot.R)),

                    (-2/delta_t)*(alpha_x*alpha_x*(position[0]-target[0])*delta_t*self.robot.r*math.cos(position[2])
                    +alpha_y*alpha_y*(position[1]-target[1])*delta_t*self.robot.r*math.sin(position[2])
                    +alpha_teta*alpha_teta*(position[2]-target[2]-theta_s(position[0], position[1]))*delta_t*self.robot.r/(2*self.robot.R))
                    ]
                
                #.r rayon des roues
                #.R demi distance entre les deux roues

                # The two args after grad are the gradient learning steps for t+1 and t
                # si critere augmente on BP un bruit fction randon_update, sion on BP le gradient
                
                if (crit_ap <= crit_av) : #amélioration
                    self.network.backPropagate(grad, 0.9,0) # grad, pas d'app, moment : permet de lisser la trajectoire
                    #ICI GRAD DOIT ETRE LA MOYENNE DES GRADIENTS CALCULES
                else : 
                    #self.network.random_update(0.001)  #propagaion d'un bruit mais cela peut éloigner le robot, peut fonctionner avec un dosage du bruit très faible
                    self.network.backPropagate(grad, 0.9, 0)
                
        self.robot.set_motor_velocity([0,0]) # stop  apres arret  du prog d'app
        #position = self.robot.get_position() #  obtient nvlle pos robot instant t+1
                #Teta_t=position[2]
             
                
        
        self.running = False       