from droidlet.lowlevel.minecraft.pyworld.item import GettableItem


class Opt:
    def __init__(self):
        self.SL = 17 * 3
        self.H = 13 * 3
        self.GROUND_DEPTH = 5
        self.mob_config = "num_mobs:4"
        # FIXME make these consts
        self.MAX_NUM_SHAPES = 3
        self.MAX_NUM_GROUND_HOLES = 3
        self.fence = True
        self.extra_simple = False
        # TODO?
        self.cuberite_x_offset = 0
        self.cuberite_y_offset = 0
        self.cuberite_z_offset = 0
        self.iglu_scenes = ""
        # FIXME! put in the scene spec generator
        self.gettable_items = [GettableItem("ball")]


opts = Opt()
