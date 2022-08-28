"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import habitat_sim
import magnum as mn
import numpy as np
import os
import quaternion


def reconfigure_scene(env, scene_path, add_humans):
    """This function takes a habitat scene and adds relevant changes to the
    scene that make it useful for locobot assistant.

    For example, it sets initial positions to be deterministic, and it
    adds some humans to the environment a reasonable places
    """

    sim = env._robot.base.sim

    # these are useful to know to understand co-ordinate normalization
    # and where to place objects to be within bounds. They change wildly
    # from scene to scene
    print("scene bounds: {}".format(sim.pathfinder.get_bounds()))

    agent = sim.get_agent(0)

    # keep the initial rotation consistent across runs
    old_agent_state = agent.get_state()
    new_agent_state = habitat_sim.AgentState()

    scene_name = os.path.basename(scene_path).split(".")[0]
    if scene_name == "mesh_semantic":
        # this must be Replica Dataset
        scene_name = os.path.basename(os.path.dirname(os.path.dirname(scene_path)))

    ###########################
    # start position
    ###########################

    # Coordinates correspond to (vertical, height, horizontal) in dashboard

    if scene_name == "apartment_0":
        # first scene in Replica Dataset
        start_position = np.asarray([0.18430093, -1.3747652, 5.265953])
        start_rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)

    elif scene_name == "devendra-home-scan":
        # Devendra's apartment

        # chair2
        # start_position = np.asarray([3.5, 0.0, -9])
        # start_rotation = quaternion.from_euler_angles(0, np.pi * (5 / 4), 0)

        # bed1, plant1, toilet1
        # start_position = np.asarray([7.5, 0.0, -9])
        # start_rotation = quaternion.from_euler_angles(0, np.pi, 0)

        # chair1
        # start_position = np.asarray([-3.31, 0., -5.11])
        # start_rotation = quaternion.from_euler_angles(0, np.pi / 2, 0)

        # couch1
        start_position = np.asarray([-2.69, 0.0, -6.96])
        start_rotation = quaternion.from_euler_angles(0, 0.52, 0)

        # couch2
        # start_position = np.asarray([7.52, 0.0, -6.23])
        # start_rotation = quaternion.from_euler_angles(0, 3.05, 0)

        # plant2
        # start_position = np.asarray([5.45, 0., -7.0])
        # start_rotation = quaternion.from_euler_angles(0, -1.04, 0)

        # toilet2
        # start_position = np.asarray([3.11, 0.0, -2.44])
        # start_rotation = quaternion.from_euler_angles(0, -0.52, 0)

        # tv1
        # start_position = np.asarray([5.26, 0., -6.60])
        # start_rotation = quaternion.from_euler_angles(0, -1.57, 0)

    elif scene_name == "fremont-home-scan":
        # Fremont space
        start_position = np.asarray([0.0, 0.0, 0.0])
        start_rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)

    else:
        # default to random navigable point
        start_position = sim.pathfinder.get_random_navigable_point()
        start_rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)
        attempt = 1
        while sim.pathfinder.distance_to_closest_obstacle(start_position) < 1.0 and attempt < 50:
            start_position = sim.pathfinder.get_random_navigable_point()
            attempt += 1

    new_agent_state.position = start_position
    new_agent_state.rotation = start_rotation
    agent.set_state(new_agent_state, reset_sensors=True, infer_sensor_states=True, is_initial=True)
    env._robot.base.init_state = agent.get_state()

    ###########################
    # scene-specific additions
    ###########################

    supported_scenes = ["skokloster-castle", "van-gogh-room", "apartment_0"]
    if scene_name not in supported_scenes:
        print("Scene {} not in supported scenes, so skipping adding objects".format(scene_name))
        return

    assets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tests/test_assets"))

    if hasattr(sim, "get_object_template_manager"):
        load_object_configs = sim.get_object_template_manager().load_object_configs
    else:
        load_object_configs = sim.load_object_configs
    human_male_template_id = load_object_configs(os.path.join(assets_path, "human_male"))[0]
    human_female_template_id = load_object_configs(os.path.join(assets_path, "human_female"))[0]

    if add_humans is True:
        if scene_name == "apartment_0":
            id_male = sim.add_object(human_male_template_id)
            id_female = sim.add_object(human_female_template_id)
            print("id_male, id_female: {} {}".format(id_male, id_female))

            sim.set_translation([1.2, -0.81, 0.3], id_female)  # apartment_0, female
            sim.set_translation([1.2, -0.75, -0.3], id_male)  # apartment_0, male

            rot = mn.Quaternion.rotation(mn.Deg(-75), mn.Vector3.y_axis())
            sim.set_rotation(rot, id_male)  # apartment_0
            sim.set_rotation(rot, id_female)  # apartment_0
        elif scene_name == "skokloster-castle":
            id_female = sim.add_object(human_female_template_id)
            print("id_female: {}".format(id_female))
            sim.set_translation([2.0, 3.0, 15.00], id_female)  # skokloster castle
        elif scene_name == "van-gogh-room":
            id_female = sim.add_object(human_female_template_id)
            print("id_female: {}".format(id_female))
            sim.set_translation([1.0, 0.84, 0.00], id_female)  # van-gogh-room

        # make the objects STATIC so that collisions work
        for obj in [id_male, id_female]:
            sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, obj)

    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings, include_static_objects=True)
