from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import copy
import time
import os

INTERPOLATE_NUM = 500
DEFAULT_PLANNING_TIME = 5.0

class PinRobot:
    def __init__(self, urdf_path, srdf_path, model_dir, viz=False) -> None:
        # Load model, visualization model, and collision model
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.geom_model = pin.buildGeomFromUrdf(self.model, urdf_path, pin.GeometryType.COLLISION, model_dir)
        self.viz_model = pin.buildGeomFromUrdf(self.model, urdf_path, pin.GeometryType.VISUAL, model_dir)

        # Add collisition pairs
        self.geom_model.addAllCollisionPairs()
        # Remove collision pairs listed in the SRDF file
        pin.removeCollisionPairs(self.model,self.geom_model,srdf_path)
        print("num collision pairs ",len(self.geom_model.collisionPairs))
        
        # Create data structures
        self.data = self.model.createData()
        self.geom_data = pin.GeometryData(self.geom_model)
        self.visual_data = pin.GeometryData(self.viz_model)

        self.get_joint_bounds()
        self.reset()
        self.viz = viz
        if self.viz:
            self.visualizer = MeshcatVisualizer(self.model, self.geom_model, self.viz_model)
            self.visualizer.initViewer()
            self.visualizer.loadViewerModel()
            self.visualizer.display(self.state)
        
    def get_joint_bounds(self):
        '''
        Get the joint bounds for the robot model
        '''
        self.joint_bounds = []
        self.num_dim = len(self.model.joints) - 1  # Exclude the universe joint
        print("Number of joints:", self.num_dim)
        for joint_id in range(1, len(self.model.joints)):
            joint_name = self.model.names[joint_id]
            joint = self.model.joints[joint_id]
            lower = self.model.lowerPositionLimit[joint.idx_q]
            upper = self.model.upperPositionLimit[joint.idx_q]
            if lower < upper:
                self.joint_bounds.append([lower, upper])
                print(f"Joint: {joint_name} - Lower: {lower}, Upper: {upper}")
        return self.joint_bounds
    
    def check_collision(self, state):
        '''
        Check if the configuration q is in collision
        '''
        state = [state[i] for i in range(self.num_dim)] #ompl state covert
        pin.computeCollisions(self.model, self.data, self.geom_model, self.geom_data, np.array(state), False)
        for k in range(len(self.geom_model.collisionPairs)):
            cr = self.geom_data.collisionResults[k]
            # cp = self.geom_model.collisionPairs[k]
            if cr.isCollision():
                # print("Collision detected in pair:", cp.first,",",cp.second)
                return True
        return False


    def get_cur_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state):
        self.state = np.array(state)
        pin.forwardKinematics(self.model, self.data, self.state)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(self.model,self.data,self.geom_model,self.geom_data,self.state)
        pin.updateGeometryPlacements(self.model,self.data, self.viz_model, self.visual_data, self.state)
        if self.viz:
            self.visualizer.display(self.state)
    
    def execute(self, path, dt=0.01):
        for state in path:
            self.set_state(state)
            time.sleep(dt)


    def reset(self):
        self.state = np.zeros(self.num_dim)

class PbStateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler

class PinOMPL:
    def __init__(self, robot) -> None:
        self.robot = robot
        # self.obstacles = obstacles

        self.space = PbStateSpace(robot.num_dim)

        bounds = ob.RealVectorBounds(robot.num_dim)
        for i, bound in enumerate(self.robot.joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))
        self.si = self.ss.getSpaceInformation()
        self.set_planner("RRT")

    def is_state_valid(self, state):
        if self.robot.check_collision(state):
            return False
        else:
            return True

    def set_planner(self, planner_name):
        '''
        Note: Add your planner here!!
        '''
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        self.ss.setPlanner(self.planner)

    def plan_start_goal(self, start, goal, allowed_time=DEFAULT_PLANNING_TIME):
        print("Planning with", self.planner.params())
        orig_state = self.robot.get_cur_state()

        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i]
            g[i] = goal[i]
        self.ss.setStartAndGoalStates(s, g)

        solved = self.ss.solve(allowed_time)
        path_list = []
        if solved:
            print(f"Found solution: interpolating into {INTERPOLATE_NUM} segments")
            path = self.ss.getSolutionPath()
            path.interpolate(INTERPOLATE_NUM)
            path_list = [self.state_to_list(state) for state in path.getStates()]
            for state in path_list:
                self.is_state_valid(state)
            success = True
        else:
            print("No solution found")
            success = False

        self.robot.set_state(orig_state)
        return success, path_list

    def plan(self, goal, allowed_time=DEFAULT_PLANNING_TIME):
        return self.plan_start_goal(self.robot.get_cur_state(), goal, allowed_time)

    def set_state_sampler(self, sampler):
        self.space.set_state_sampler(sampler)

    def state_to_list(self, state):
        return [state[i] for i in range(self.robot.num_dim)]
    

if __name__ == '__main__':


    root_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(root_dir, "models", "fr3_dual.urdf")
    srdf_path = os.path.join(root_dir, "models", "fr3_dual.srdf")
    model_dir = os.path.join(root_dir, "models")
    start_state =  [0, -0.778, 0.0158,-2.369, 0, 1.54, 0.77, 0, 0, 0, -0.778, 0.0158,-2.369, 0, 1.54, 0.77, 0, 0]
    goal_state = copy.deepcopy(start_state)
    goal_state[1] += 0.5
    goal_state[10] += 0.5

    robot = PinRobot(urdf_path, srdf_path, model_dir, viz=True)
    pb_ompl_interface = PinOMPL(robot)
    pb_ompl_interface.set_planner("bitstar")

    pb_ompl_interface.robot.set_state(start_state)
    res, path  = pb_ompl_interface.plan(goal_state)
    if res:
        pb_ompl_interface.robot.execute(path)

        
        