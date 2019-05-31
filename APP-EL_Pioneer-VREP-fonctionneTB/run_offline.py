# coding: utf-8

from BackProp_Python_v2 import NN
from vrep_pioneer_simulation import VrepPioneerSimulation
from rdn import Pioneer  # rdn pour ROS avec le pioneer
# import rospy
from online_trainer import OnlineTrainer
from offline_trainer import OfflineTrainer
import json
import threading




HL_size = 9
# nbre neurons of Hiden layer
network = NN(3, HL_size, 2)

robot = VrepPioneerSimulation()
trainer = OfflineTrainer(robot, network)
target = [0., 0., 0.]

trainer.training = True


continue_running = True
while (continue_running):

    thread = threading.Thread(target=trainer.train, args=(target,))
    trainer.running = True
    thread.start()
    trainer.running = False
    continue_running = False


json_obj = {"input_weights": network.wi, "output_weights": network.wo}
with open('last_w.json', 'w') as fp:
    json.dump(json_obj, fp)

print("The last weights have been stored in last_w.json")
