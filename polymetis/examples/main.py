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
    current_joint = torch.Tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
        # Move with a constant value
        const_val = 0.3
        delta_joint_pos_desired = torch.mul(const_val,self.current_joint)
        joint_pos_desired = joint_pos + delta_joint_pos_desired
        state_log = self.robot.set_joint_positions(joint_pos_desired, time_to_go=2.0)  
        # Get updated joint positions
        joint_pos = self.robot.get_joint_angles()
        self.ids.robot_status.text = f"NEW JOINT POSITIONS: {joint_pos}"  

    def get_ee_pos(self):
        # Get ee pose
        ee_pos, ee_quat = self.robot.pose_ee()
        self.ids.status_ee_pos.text = f"Current ee position: {ee_pos}"

    def spinner_clicked(self,value):
        if value == "Joint #1":
            self.current_joint = torch.Tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif value == "Joint #2":
            self.current_joint = torch.Tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        elif value == "Joint #3":
            self.current_joint = torch.Tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])                    
        elif value == "Joint #4":
            self.current_joint = torch.Tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        elif value == "Joint #5":
            self.current_joint = torch.Tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])   
        elif value == "Joint #6":
            self.current_joint = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        else:
            #no joint selected, do not move any joint 
            current_joint = torch.Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])   

class RootWidget(ScreenManager):
    pass

class MainApp (App):
    def build(self):
        return RootWidget()

if __name__ == "__main__":

    MainApp().run()

