import time
import math

L = 3 #demi-longueur du carr� explorable
i,j,k = 1,1,1 #coefficients de discr�tisations 
theta = 0
M = 1 #vitesse angulaire max des roues en sortie
position= [0,0,0]
pi = math.pi
sample_position = [] #liste des positions cibles (ds le rep�re du robot) finales
sample_command = [] #liste des commandes des roues finales

#g�n�ration de la base de test normalisee
#boucle pour g�n�rer les combinaisons 1 � 6
for i in [1/4, 1/2, 3/4, 1] :
    for j in [1/4, 1/2, 3/4, 1] :
        for theta in [-pi ,-(3*pi)/4,-pi/2,-pi/4,0,pi/4,pi/2,3*pi/4] :
            sample_position += [[1/i,0,theta/(2*pi)],#1
            [1*i,1*j,theta/(2*pi)],#2
            [-1*i,1*j,theta/(2*pi)],#3
            [-1*i,0,theta/(2*pi)],#4
            [-1*i,-1*j,theta/(2*pi)],#5
            [1*i,-1*j,theta/(2*pi)]]#6
            
            sample_command += [[1,1],#1
            [-1,1],#2
            [1,-1],#3
            [-1,-1],#4
            [-1,1],#5
            [1,-1]]#6
            
#boucle pour g�n�rer les combinaisons 7 et 8
for k in [(3/4),(1/2),(1/4)]:
    sample_position += [[0,0,-1*(pi*k)/(2*pi)],#7 
    [0,0,pi*k/(2*pi)]]#8
    
    sample_command += [[1,-1],#7
    [-1,1]]#8

#g�n�ration du point final   #9
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

#liste des sorties d�sir�es
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

        self.alpha = [1/(2*L),1/(2*L),1/(math.pi)]  # normalition avec limite du monde cartesien = -3m � + 3m
        #pourquoi pas avec 1/L ? sinon on va de -0.5 � 0.5.

	#il faut que quand training soit True l'apprentissage soit lanc� d'un bloc. Il faut qu'il soit False qd le robot bouge
    def train(self, target):
        somme_erreur_av = len(sample_position) * 8 #erreur maximum 


		while self.training:
			
			#calcul de l'erreur
			somme_erreur = 0
			for i in range(len(sample_position)) :
				position_ponderee = [sample_position[i]/alpha[i] for i in range(3)]
				command = self.network.runNN(sample_position[i]) # propage erreur et calcul vitesses roues instant t
				erreur = (command[0]-sample_command[i][0])**2 + (command[1]-sample_command[i][1])**2
				somme_erreur += erreur

			grad = somme_erreur / (1.0*len(sample_position)) # erreur moyenne

			if somme_erreur < somme_erreur_av :
					self.network.backPropagate(grad, 0.9,0) # grad, pas d'app, moment : permet de lisser la trajectoire
					somme_erreur_av = somme_erreur
			else :
				self.training = false


        while self.running:
            debut = time.time()
            command = self.network.runNN([0,0,0]) # propage erreur et calcul vitesses roues instant t
            
            #pond�ration des erreurs           
            alpha_x = 1/(2*L)
            alpha_y = 1/(2*L)
            alpha_teta = 1.0/(math.pi)
            
                       
            self.robot.set_motor_velocity(command) # applique vitesses roues instant t,                     
            time.sleep(0.050) # attend delta t
            position = self.robot.get_position() #  obtient nvlle pos robot instant t+1       
            
            network_input[0] = (position[0]-target[0])*self.alpha[0]
            network_input[1] = (position[1]-target[1])*self.alpha[1]
            network_input[2] = (position[2]-target[2]-theta_s(position[0], position[1]))*self.alpha[2]

      
        self.robot.set_motor_velocity([0,0]) # stop  apres arret  du prog d'app
        #position = self.robot.get_position() #  obtient nvlle pos robot instant t+1
                #Teta_t=position[2]
 
        self.running = False       