"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from droidlet.dialog.dialogue_manager import DialogueManager
from droidlet.memory.memory_nodes import ChatNode, PlayerNode, ProgramNode


class SwarmDialogueManager(DialogueManager):
    def __init__(self, memory, dialogue_object_classes, opts, low_level_interpreter_data={}):
        super(SwarmDialogueManager, self).__init__(
            memory, dialogue_object_classes, opts, low_level_interpreter_data
        )

    def neglect(self, name):
        if "bot" in name:
            return True

    def get_last_m_chats(self, m=1):
        # fetch last m chats from memory
        all_chats = self.memory.nodes[ChatNode.NODE_TYPE].get_recent_chats(self.memory, n=m)
        chat_list_text = []
        for chat in all_chats:
            # does not need to interpret its own swarm chats
            speaker = self.memory.nodes[PlayerNode.NODE_TYPE](self.memory, chat.speaker_id).name
            if self.neglect(speaker):
                continue
            chat_memid = chat.memid
            # get logical form if any else None
            logical_form, chat_status = None, ""
            logical_form_triples = self.memory.get_triples(
                subj=chat_memid, pred_text="has_logical_form"
            )
            processed_status = self.memory.get_triples(
                subj=chat_memid, pred_text="has_tag", obj_text="unprocessed"
            )
            if logical_form_triples:
                logical_form = self.memory.nodes[ProgramNode.NODE_TYPE](
                    self.memory, logical_form_triples[0][2]
                ).logical_form

            if processed_status:
                chat_status = processed_status[0][2]
            chat_str = chat.chat_text
            chat_list_text.append((speaker, chat_str, logical_form, chat_status, chat_memid))

        return chat_list_text
