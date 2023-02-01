from threading import Thread

from droidlet.lowlevel.minecraft.pyworld.world import World
from droidlet.lowlevel.minecraft.pyworld.fake_mobs import SimpleMob, make_mob_opts
from droidlet.lowlevel.minecraft.pyworld.ticker import Ticker
from droidlet.lowlevel.minecraft.small_scenes_with_shapes import build_shape_scene
from droidlet.lowlevel.minecraft.pyworld.world_config import opts


class Opt:
    pass


def instantiate_world_from_spec(opts):
    scene_spec = build_shape_scene(opts)

    def block_generator(world):
        world.blocks[:] = 0
        for b in scene_spec["schematic_for_cuberite"]:
            x, y, z = world.to_npy_coords((b["x"], b["y"], b["z"]))
            # TODO maybe don't just eat every error, do this more carefully
            try:
                world.blocks[x, y, z][0] = b["id"]
                world.blocks[x, y, z][1] = b["meta"]
            except:
                pass

    world_opts = Opt()
    world_opts.sl = opts.SL
    world_opts.world_server = True
    world_opts.port = 6002
    mobs = []
    for mob_spec in scene_spec["mobs"]:
        # FIXME add more possibilities:
        assert mob_spec["mobtype"] in ["rabbit", "cow", "pig", "chicken", "sheep"]
        mob_opt = make_mob_opts(mob_spec["mobtype"])
        x, y, z, pitch, yaw = mob_spec["pose"]
        mobs.append(SimpleMob(mob_opt, start_pos=(x, y, z), start_look=(yaw, pitch)))
    # FIXME get this from the scene generator
    items = getattr(opts, "gettable_items", [])
    world = World(
        world_opts,
        {
            "ground_generator": block_generator,
            "mobs": mobs,
            "players": [],
            "agent": None,
            "items": items,
        },
    )
    return world


if __name__ == "__main__":
    ticker = Ticker(tick_rate=0.01, step_rate=0.2, ip="localhost", port=6002)
    ticker_thread = Thread(target=ticker.start, args=())
    ticker_thread.start()

    world = instantiate_world_from_spec(opts)
