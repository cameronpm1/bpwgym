from dm_control import mujoco

physics = mujoco.Physics.from_xml_string('./urdf/bluebody-urdf.xml')
# Render the default camera view as a numpy array of pixels.
pixels = physics.render()

# Reset the simulation, move the slide joint upwards and recompute derived
# quantities (e.g. the positions of the body and geoms).
with physics.reset_context():
  physics.named.data.qpos['up_down'] = 0.5

# Print the positions of the geoms.
print(physics.named.data.geom_xpos)
