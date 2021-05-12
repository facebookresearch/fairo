from base_agent.memory_nodes import PlayerNode


class HeuristicPerception:
    def __init__(self, agent):
        self.agent = agent

    def perceive(self):
        bots = self.agent.world.get_bots()
        for bot in bots:
            #            print(f"[Perception INFO]: Perceived bot [{bot.name}] in the world, update in memory]")
            bot_node = self.agent.memory.get_player_by_eid(bot.entityId)
            if bot_node is None:
                memid = PlayerNode.create(self.agent.memory, bot)
                bot_node = PlayerNode(self.agent.memory, memid)
                self.agent.memory.tag(memid, "bot")
            bot_node.update(self.agent.memory, bot, bot_node.memid)
            print(
                f"[Memory INFO]: update bot [{bot.name}] position: ({bot.pos.x}, {bot.pos.y}, {bot.pos.z})"
            )

        bot_memids = self.agent.memory.get_memids_by_tag("bot")
        bots_in_world = [b.entityId for b in bots]
        for memid in bot_memids:
            bot_eid = self.agent.memory.get_mem_by_id(memid).eid
            if bot_eid not in bots_in_world:
                self.agent.memory.forget(memid)
                print(f"[Memory INFO]: delete bot [{bot_eid}] from memory")
