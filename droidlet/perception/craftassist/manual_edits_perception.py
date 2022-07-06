"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

class ManualEditsPerception:
    """Perceive the world at a given frequency and send updates back to the agent

    updates attributes of all spatial objects,
    takes this information from manual changes made in frontend

    Args:
        agent (LocoMCAgent): reference to the minecraft Agent
        perceive_freq (int): if not forced, how many Agent steps between perception
    """

    def __init__(self, agent, perceive_freq=5):
        self.agent = agent
        self.memory = agent.memory  # NOTE: remove this once done
        self.perceive_freq = perceive_freq
        self.edits = {} # dictionary with (memid: <dictionary_of_edits>) k-v pairs

    def perceive(self, force=False):
        """
        Every n agent_steps (defined by perceive_freq), update in agent memory
        all manually changed objects (done via dashboard map).

        Args:
            force (boolean): set to True to run all perceptual heuristics right now,
                as opposed to waiting for perceive_freq steps (default: False)
        """
        
        if self.edits:
            for memid in self.edits.keys():
                toEdit = {attr: val for attr, val in self.edits[memid].items() if attr not in ("location", "pos")}
                if toEdit:
                    cmd = "UPDATE ReferenceObjects SET " + "=?, ".join(toEdit.keys()) + "=? WHERE uuid=?"
                    self.memory.db_write(
                        cmd, 
                        *toEdit.values(), 
                        memid
                    )

                if "pos" in self.edits[memid].keys():
                    # spatial data is iterable, needs to be handled differently
                    newPos = self.edits[memid]["pos"]
                    assert len(newPos) == 3
                    cmd = "UPDATE ReferenceObjects SET x=?, y=?, z=? WHERE uuid=?"
                    self.memory.db_write(
                        cmd,
                        newPos[0], newPos[1], newPos[2],
                        memid
                    )

        return

    def add_edit(self, new_edit):
        """
        Takes flattened object input `new_edit` and unflattens using "memid" as key.
        Then store/update unflattened object in perception module

        Args:
            new_edit (dict): Flattened mapping in form of
                "memid": {"value": <memid>, "valueType": "string",  "status": "same"},
                <attr1>: {"value": <new_value>, ...,                "status": "changed"},
                <attr2>: {"value": <new_value>, ...,                "status": "same"},
                ...
        """

        memid = new_edit["memid"]["value"]
        if memid not in self.edits:
            self.edits[memid] = {}
        for attr in new_edit.keys():
            if new_edit[attr]["status"] == "changed":
                self.edits[memid][attr] = new_edit[attr]["value"]
