import Pyro4
from polymetis import RobotInterface
import torch

Pyro4.config.SERIALIZERS_ACCEPTED.add("pickle")
Pyro4.config.ITER_STREAMING = True
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

@Pyro4.expose
class RemoteFranka(object):
    def __init__(self):
        self.robot = RobotInterface ( ip_address = "localhost")

    def connect(self):
        if self.robot == None:
            self.robot = RobotInterface ( ip_address = "localhost")
        
    def go_home(self):
        #Reset
        self.robot.go_home()
        joint_pos = self.robot.get_joint_angles()

    def move(self, joint,vel):
        # Get joint positions
        joint_pos = self.robot.get_joint_angles()
        if joint == 1:
        	delta_joint_pos_desired = torch.Tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])	
        elif joint == 2:	
        	delta_joint_pos_desired = torch.Tensor([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif joint == 3:	
        	delta_joint_pos_desired = torch.Tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        elif joint == 4:	
        	delta_joint_pos_desired = torch.Tensor([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        elif joint == 5:	
        	delta_joint_pos_desired = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        elif joint == 6:	
        	delta_joint_pos_desired = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0])
        else:	
        	delta_joint_pos_desired = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        	        	        	        	        	        	
        joint_pos_desired = joint_pos + delta_joint_pos_desired
        state_log = self.robot.set_joint_positions(joint_pos_desired, time_to_go=(4.0-vel))  
        # Get updated joint positions
        joint_pos = self.robot.get_joint_angles()
                                     
    def get_ee_pos(self):
        # Get ee pose
        ee_pos, ee_quat = self.robot.pose_ee()
        pos = ee_pos.numpy()
        return pos

    def go_to_ee_pos(self, pos):

        # Command robot to ee pose (move ee downwards)
        # note: can also be done with robot.move_ee_xyz
        ee_pos, ee_quat = self.robot.pose_ee()
        axis =torch.Tensor([0.0, 0.0, -1])
        const = float(pos)
        #print (const)
        delta_ee_pos_desired = torch.mul(const,axis)
        #print(delta_ee_pos_desired)
        ee_pos_desired = ee_pos + delta_ee_pos_desired
        self.robot.set_ee_pose( position=ee_pos_desired, orientation=None, time_to_go=2.0)
    
    def print_franka(self):
        print ("Connected to Franka")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pass in server device IP")
    parser.add_argument(
        "--ip",
        help="Server device (robot) IP. Default is 192.168.0.0",
        type=str,
        default="192.168.0.0",
    )

    args = parser.parse_args()
    
    print (f'IP:: >  {args.ip}')

    # GLContexts in general are thread local
    # The PyRobot <-> Habitat integration is not thread-aware / thread-configurable,
    # so our only option is to disable Pyro4's threading, and instead switch to
    # multiplexing (which isn't too bad)
    Pyro4.config.SERVERTYPE = "multiplex"
    robot = RemoteFranka()
    
    
    
    #daemon = Pyro4.Daemon(args.ip)
    #daemon = Pyro4.Daemon()
    print ("Deamen server created")
    #ns = Pyro4.locateNS()
    
    #robot_uri = daemon.register(robot)
    
    daemon = Pyro4.Daemon.serveSimple({robot: 'remotefranka',}, host=args.ip, port=9090, ns=False, verbose=True)

    daemon.close()
    print('daemon closed')
    
    #ns.register("remotelocobot", robot_uri)

    #print("Server is started...")
    #daemon.requestLoop()


# Below is client code to run in a separate Python shell...
# import Pyro4
# robot = Pyro4.Proxy("PYRONAME:remotelocobot")
# robot.go_home()
