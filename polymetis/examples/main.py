from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.config import Config
#Setting kivy window size
Config.set('graphics','width','1200')
Config.set('graphics', 'height', '600')

import torch
from polymetis import RobotInterface

Builder.load_file('design.kv')

class RobotControlScreen(Screen):

    robot = RobotInterface ( ip_address = "localhost")

    def connect(self):
        self.ids.robot_status.text = "Connecting"
        if self.robot == None:
            self.robot = RobotInterface ( ip_address = "localhost")
        self.ids.robot_status.text = "Connected"
        
    def go_home(self):
        self.ids.robot_status.text = "Going Home"
        #Reset
        self.robot.go_home()
        joint_pos = self.robot.get_joint_angles()
        self.ids.robot_status.text = f"At Home Pos: {joint_pos}"

    def move(self):
        self.ids.robot_status.text = "Moving"
        # Get joint positions
        joint_pos = self.robot.get_joint_angles()
        delta_joint_pos_desired = torch.Tensor([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        joint_pos_desired = joint_pos + delta_joint_pos_desired
        state_log = self.robot.set_joint_positions(joint_pos_desired, time_to_go=2.0)  
        # Get updated joint positions
        joint_pos = self.robot.get_joint_angles()
        self.ids.robot_status.text = f"NEW JOINT POSITIONS: {joint_pos}"  

    def get_ee_pos(self):
        # Get ee pose
        ee_pos, ee_quat = self.robot.pose_ee()
        self.ids.status_ee_pos.text = f"Current ee position: {ee_pos}"

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

class RootWidget(ScreenManager):
    pass

class MainApp (App):
    def build(self):
        return RootWidget()

if __name__ == "__main__":

    MainApp().run()

