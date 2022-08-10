OS_ENV_DICT = {
    "AWS_ACCESS_KEY_ID": "test_key_id",
    "AWS_SECRET_ACCESS_KEY": "secretkkkkkk",
    "AWS_DEFAULT_REGION": "us-west-1"
}

class S3ClientMock:
    def __init__(self) -> None:
        self.method_called = {}

    def _add_method_called_count(self, method_name):
        if method_name not in self.method_called:
            self.method_called[method_name] = 0
        self.method_called[method_name] += 1

class MockListObjResult:
    def __init__(self) -> None:
        print("init")
        pass
    
    def search(search_param = None):
        print(search_param)
        return [
            {"Prefix": "20280224132033/"}, 
            {"Prefix": "20290224132013/"}, 
        ]

def mock_make_api_call(caller, method_name, params):
    if method_name == "ListObjects":
        print("list obj")
        # called list_objects
        return MockListObjResult()

        
        


        