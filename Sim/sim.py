""" Base class for simulation """

from collections import namedtuple
from typing import Callable, Union, Dict, List, Optional

import glfw
import mujoco as mj
import numpy as np


class Sim:

    """
    Base class that wraps around MuJoCo simulation objects and implements commonly used fuctionality
    such as stepping, rendering and callbacks.
    """

    def __init__(
        self,
        model: str,
        dt: float,
    ) -> None:
        """
        Simulation class for loading and setting up mujoco xml file
        model -> file location
        dt -> length of timestep (must be greater than or equal to dt in the xml file)
        """
        self._model = mj.MjModel.from_xml_path(model)
        self._step = 0
        self._data = None
        self._state = None
        self._labels = None
        self._initial_state = None

        self.set_data()
        self._initial_state = self._data
        
        self._dt = self.dt()
        self._nstep = int(np.ceil(dt / self._dt))

        # Rendering
        self._print_camera_config = 0 
        self._window = None
        self._scene = None
        self._context = None
        self._cam = None
        self._opt = None
        self._button_left = False
        self._button_middle = False
        self._button_right = False
        self._lastx = 0
        self._lasty = 0

    def dt(self) -> None:
        """Return simulation timestep"""
        return self._model.opt.timestep 

    def model(self):
        return self._model

    def set_data(self):
        self._data = mj.MjData(self._model)

    def advance(self, dt: float) -> None:
        """
        Advance the simulation by dt
        """
        mj.mj_step(self._model, self._data, nstep=self._nstep)
        self._step += 1

    def step(self) -> None:
        """Step the simulation by dt"""
        self.advance(self.dt)

    def set_ctrl(self, ctrl: np.ndarray = None):
        """Set control"""
        self._data.ctrl[:] = ctrl[:]

    def reset(self):
        mj.mj_resetData(self._model, self._initial_state)
        return self.get_state()

    def set_state(self,
        joints: int,
        jnames: List[str] = None,
    ) -> dict:
        """
        Set state dictionary for simulation class, requires the number of joints 
        in the xml file, can also take a list of joint names (ordered correctly)
        """
        if jnames is not None and joints != len(jnames):
            print('Error: joint names list is not the same length as number of joints')
            quit()
        self._state = {}
        self._labels = {}
        qpos_indx = self._model.jnt_qposadr
        qvel_indx = self._model.dof_jntid
        for i in range(joints):
            if i < joints-1:
                pos = range(qpos_indx[i],qpos_indx[i+1])
            else:
                pos = range(qpos_indx[i],len(self._data.qpos))
            vel = np.where(qvel_indx == i)[0]
            if jnames is None:
                name = str(i)
            else:
                name = jnames[i]
            self._labels[name] = [pos,vel]
            self._state[name] = {'qpos':np.take(self._data.qpos,pos),'qvel':np.take(self._data.qvel,vel)}
        self._state['ctrl'] = self._data.ctrl

        return self._state

    def get_state(self) -> dict:
        """
        update and return the state dictionary variable
        """
        if self._state is None:
            print('Error: state has not been set')
            quit()
        #self.set_data()
        for label in self._labels.keys():
            for i,indx in enumerate(self._labels[label][0]):
                self._state[label]['qpos'][i] = self._data.qpos[indx]
            for i,indx in enumerate(self._labels[label][1]):
                self._state[label]['qvel'][i] = self._data.qvel[indx]
        self._state['ctrl'] = self._data.ctrl

        return self._state

    

    '''
     For rendering
    '''

    def render(self):
        if glfw.window_should_close(self._window):
            self.terminate()
            quit()

        viewport_width, viewport_height = glfw.get_framebuffer_size(self._window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        #print camera configuration (help to initialize the view)
        if (self._print_camera_config==1):
            print('cam.azimuth =',self._cam.azimuth,';','cam.elevation =',self._cam.elevation,';','cam.distance = ',self._cam.distance)
            print('cam.lookat =np.array([',self._cam.lookat[0],',',self._cam.lookat[1],',',self._cam.lookat[2],'])')

        # Update scene and render
        mj.mjv_updateScene(self._model, self._data, self._opt, None, self._cam, mj.mjtCatBit.mjCAT_ALL.value, self._scene)
        mj.mjr_render(viewport, self._scene, self._context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self._window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()


    def set_up_screen(self): 
        self._cam = mj.MjvCamera()                        # Abstract camera
        self._opt = mj.MjvOption()                        # visualization options

        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        self._window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self._window)
        glfw.swap_interval(1)

        # initialize visualization data structures
        mj.mjv_defaultCamera(self._cam)
        mj.mjv_defaultOption(self._opt)
        self._scene = mj.MjvScene(self._model, maxgeom=10000)
        self._context = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # install GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self._window, self.keyboard)
        glfw.set_cursor_pos_callback(self._window, self.mouse_move)
        glfw.set_mouse_button_callback(self._window, self.mouse_button)
        glfw.set_scroll_callback(self._window, self.scroll)

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self._model, self._data)
            mj.mj_forward(self._model, self._data)

    def mouse_button(self, window, button, act, mods):

        self._button_left = (glfw.get_mouse_button(
            self._window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self._button_middle = (glfw.get_mouse_button(
            self._window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self._button_right = (glfw.get_mouse_button(
            self._window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        # update mouse position
        glfw.get_cursor_pos(self._window)

    def mouse_move(self, window, xpos, ypos):
        # compute mouse displacement, save

        dx = xpos - self._lastx
        dy = ypos - self._lasty
        self._lastx = xpos
        self._lasty = ypos

        # no buttons down: nothing to do
        if (not self._button_left) and (not self._button_middle) and (not self._button_right):
            return

        # get current window size
        width, height = glfw.get_window_size(self._window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(
            self._window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(
            self._window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        # determine action based on mouse button
        if self._button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self._model, action, dx/height,
                        dy/height, self._scene, self._cam)

    def scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self._model, action, 0.0, -0.05 *
                        yoffset, self._scene, self._cam)

    def terminate(self):
        glfw.terminate()

def main():
    sim = Sim('/home/ubuntu/robot/robot.xml',0.05)
    print(sim._model.joint('joint011'))
    #sim.set_state(19)
    #print(sim.get_state())
    i=0
    sim.set_up_screen()
    while i < 500:
        sim.step()
        sim.render()
    sim.terminate()

if __name__ == "__main__":
    main()
