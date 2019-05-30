import time
import math

L = 3  # demi-longueur du carre explorable


def theta_s(x, y):
    if x > 0:
        return 1 * math.atan(1 * y)
    if x <= 0:
        return 1 * math.atan(-1 * y)


class OnlineTrainer:
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
        # pourquoi pas avec 1/L ? sinon on va de -0.5 � 0.5.

    def train(self, target):

        position = self.robot.get_position()

        network_input = [0, 0, 0]
        network_input[0] = (position[0] - target[0]) * self.alpha[0]
        network_input[1] = (position[1] - target[1]) * self.alpha[1]
        network_input[2] = (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]
        # Teta_t = 0

        while self.running:
            debut = time.time()
            command = self.network.runNN(network_input)  # propage erreur et calcul vitesses roues instant t

            # pond�ration des erreurs
            alpha_x = 1 / (2 * L)
            alpha_y = 1 / (2 * L)
            alpha_teta = 1.0 / (math.pi)

            # fonction de co�t avant le d�placement
            crit_av = alpha_x * alpha_x * (position[0] - target[0]) * (position[0] - target[0]) + alpha_y * alpha_y * (
                    position[1] - target[1]) * (position[1] - target[1]) + alpha_teta * alpha_teta * (
                              position[2] - target[2] - theta_s(position[0], position[1])) * (
                              position[2] - target[2] - theta_s(position[0], position[1]))

            self.robot.set_motor_velocity(command)  # applique vitesses roues instant t,
            time.sleep(0.050)  # attend delta t
            position = self.robot.get_position()  # obtient nvlle pos robot instant t+1

            network_input[0] = (position[0] - target[0]) * self.alpha[0]
            network_input[1] = (position[1] - target[1]) * self.alpha[1]
            network_input[2] = (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]

            # fonction de co�t apr�s le d�placement
            crit_ap = alpha_x * alpha_x * (position[0] - target[0]) * (position[0] - target[0]) + alpha_y * alpha_y * (
                    position[1] - target[1]) * (position[1] - target[1]) + alpha_teta * alpha_teta * (
                              position[2] - target[2] - theta_s(position[0], position[1])) * (
                              position[2] - target[2] - theta_s(position[0], position[1]))

            if self.training:
                delta_t = (time.time() - debut)

                # moyenne des erreurs
                grad = [
                    (-2 / delta_t) * (alpha_x * alpha_x * (position[0] - target[0]) * delta_t * self.robot.r * math.cos(
                        position[2])
                                      + alpha_y * alpha_y * (
                                              position[1] - target[1]) * delta_t * self.robot.r * math.sin(position[2])
                                      - alpha_teta * alpha_teta * (position[2] - target[2] - theta_s(position[0],
                                                                                                     position[
                                                                                                         1])) * delta_t * self.robot.r / (
                                              2 * self.robot.R)),

                    (-2 / delta_t) * (alpha_x * alpha_x * (position[0] - target[0]) * delta_t * self.robot.r * math.cos(
                        position[2])
                                      + alpha_y * alpha_y * (
                                              position[1] - target[1]) * delta_t * self.robot.r * math.sin(position[2])
                                      + alpha_teta * alpha_teta * (position[2] - target[2] - theta_s(position[0],
                                                                                                     position[
                                                                                                         1])) * delta_t * self.robot.r / (
                                              2 * self.robot.R))
                ]

                # .r rayon des roues
                # .R demi distance entre les deux roues

                # The two args after grad are the gradient learning steps for t+1 and t
                # si critere augmente on BP un bruit fction randon_update, sion on BP le gradient

                if (crit_ap <= crit_av):  # am�lioration
                    self.network.backPropagate(grad, 0.9,
                                               0)  # grad, pas d'app, moment : permet de lisser la trajectoire
                    # ICI GRAD DOIT ETRE LA MOYENNE DES GRADIENTS CALCULES
                else:
                    # self.network.random_update(0.001)  #propagaion d'un bruit mais cela peut �loigner le robot, peut fonctionner avec un dosage du bruit tr�s faible
                    self.network.backPropagate(grad, 0.9, 0)

        self.robot.set_motor_velocity([0, 0])  # stop  apres arret  du prog d'app
        # position = self.robot.get_position() #  obtient nvlle pos robot instant t+1
        # Teta_t=position[2]

        self.running = False
