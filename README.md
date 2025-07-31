# ompl_pinocchio
This repo provides interface to use OMPL and Pinocchio for motion planning of dual-arm Franka Fr3 robot.

![example](/images/example.png)

# Environment
Tested with:<br>
**Python 3.10**<br>
**Ubuntu20.04**

# Installation instructions:

## Create a Conda Environment 
```
conda create -n ompl python=3.10
conda activate ompl
```

## Install [OMPL](https://ompl.kavrakilab.org/installation.html) for motion planning
```
pip install ompl
```
## Install and [Pinocchio](https://github.com/stack-of-tasks/pinocchio) for kinematics and collision checking
```
conda install pinocchio -c conda-forge
```

## Install MeshCat for Visualization
```
pip install meshcat
```

# Demo
One example is provided.
This demo performs arm motion planning for a dual-arm Franka Fr3 robot.
You can view the visualization by opening the output URL in your browser.

```
python pin_ompl.py
```


# Additional Information
1. To use a different robot, make sure to provide the corresponding URDF, SRDF, and mesh files, similar to the ones found in the `models` directory.
