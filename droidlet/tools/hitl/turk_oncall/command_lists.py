HEADER = ["subdomain", "batch", "list_id", "command_list", "answer_list"]

NUM_LISTS = 17

COMMAND_LISTS = [
    "make two red cubes there|is there anything red|how many cubes are there",
    "make two red cubes|what is to the left of that (point at one of the cubes)|can you climb on top of the cube",
    "make two red cubes|how many red things are there|point at the cube",
    "make a big green square behind me|what color is the square|show me to the square",
    "make a big green square|what is the name of the thing closest to you|is there anything big",
    "make a big green square|what size is the square|destroy that (look at the square)",
    "make a green square|make a yellow circle to the left of the square|what is to the left of the square",
    "make a green square|make a yellow circle to the left of the square|where is the circle",
    "make a yellow circle|what is the name of the thing closest to me|can you topple the circle",
    "where are you|what can you do|where am I",
    "spawn two pigs|have you seen the pig|find the pig",
    "follow the sheep|dig two tiny holes there|fill that hole up with grass",
    "make two circles there|look at the circle|go to circle",
    "come here|what is your favorite object|follow me",
    "go there|turn left|turn right",
    "destroy the house|go forward 1 step|go backward",
    "hello|go left|go right",
]

ANSWER_LISTS = [
    "assistant should make two red cubes|assistant should answer yes|assistant should answer two",
    "assistant should make two red cubes|assistant answer should make sense|assistant should move to top of cube",
    "assistant should make two red cubes|assistant should answer two|assistant should flash a cube",
    "assistant should make a square|assistant should answer green|assistant should answer with the square location",
    "assistant should make a square|assistant answer should make sense|assistant should anwer yes",
    "assistant should make a square|assistant should answer big|assistant should destroy the square",
    "assistant should make a square|assistant should make a yellow circle|assistant should answer circle",
    "assistant should make a square|assistant should make a yellow circle|assistant should answer with circle location",
    "assistant should make a yellow circle|assistant answer should make sense|assistant should say they don't know how to do that",
    "assistant should answer with their location|assistant should say a capability|assistant should answer with your location",
    "assistant should spawn two pigs|assistant should answer with location of a pig|assistant should go to a pig",
    "assistant should follow the sheep if and only if there are any (may ask for clarification)|assistant should dig two holes|assistant should fill one hole",
    "assistant should make two circles|assistant should face the circle|assistant should go to the circle",
    "assistant should move to you|assistant should say there is no object marked as favorite|assistant should follow you",
    "assistant should move to where you are looking|assistant should turn left|assistant should turn right",
    "assistant should ask for clarification|assistant should move forward one step|assistant should walk backward",
    "assistant should respond with a greeting|assistant should move to your left|assistant should move to your right",
]
