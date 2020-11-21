"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
from base_sql_mock_environment import BaseSQLMockEnvironment
from typing import Dict


class SQLConverter(BaseSQLMockEnvironment):
    """A script with basic commands dealing with reference objects.
    """

    def __init__(self):
        super().__init__()
        self.all_sql_queries = []

    def run(self, action_dict):
        self.agent.memory.sql_queries = []
        self.convert_sql_query(action_dict)
        self.all_sql_queries.append(self.agent.memory.sql_queries)
        print(self.all_sql_queries)

    def convert_sql_query(self, action_dict: Dict):
        """
        Get SQL commands for an action dictionary.
        """
        print(action_dict)
        self.handle_logical_form(action_dict)

    def process_input_dataset(self):
        """
        Read annotated dataset and get action dictionaries.
        """
        action_dict = {
            "dialogue_type": "HUMAN_GIVE_COMMAND",
            "action": {
                "action_type": "DESTROY",
                "reference_object": {
                    "has_name": "sphere",
                    "has_colour": "red",
                    "has_size": "small",
                },
            },
        }
        self.run(action_dict)


def main():
    sql_converter = SQLConverter()
    sql_converter.process_input_dataset()


if __name__ == "__main__":
    main()
