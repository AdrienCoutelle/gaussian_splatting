Here is how the gaussian splatting renderer works:

First the gaussians are defined in the world space, they have to be transformed to the camera space. The stadard pose in gaussian splatting reprensents the camera-to-world transformation.

The position how the gaussians in the camera space is computed as follows:

pos_world = r_camera_to_world.pos_camera + t_camera_to_world
<=> pos_world - t_camera_to_world = r_camera_to_world.pos_camera
<=> pos_camera = r_world_to_camera.(pos_world - t_camera_to_world)

