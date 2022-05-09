HEADER = ["subdomain", "batch", "list_id", "command_list", "answer_list"]

NUM_LISTS = 6

COMMAND_LISTS = [
    "make two red cubes there|is there anything red|how many cubes are there|what is to the left of that (point at one of the cubes)|can you climb on top of the cube|how many red things are there|point at the cube",
    "make a big green square behind me|what color is the square|show me to the square|what is the name of the thing closest to you|is there anything big|what size is the square|destroy that (look at the square)",
    "make a green square|make a yellow circle to the left of the square|what is to the left of the square|where is the circle|what is the name of the thing closest to me|can you topple the circle",
    "where are you|what can you do|where am I|spawn two pigs|have you seen the pig|find the pig|follow the sheep",
    "dig two tiny holes there|fill that hole up with grass|make two circles there|look at the circle|what is your favorite object to my left|go to circle|follow me",
    "come here|go there|turn left|turn right|go forward 1 step|go backward|go left|go right",
]

ANSWER_LISTS = [
    "agent should make two red cubes|agent should answer yes|agent should answer two|agent answer should make sense|agent should move to top of cube|agent should answer two|agent should flash a cube",
    "agent should make a square|agent should answer green|agent should answer with the square location|agent answer should make sense|agent should anwer yes|agent should answer big|agent should destroy the square",
    "agent should say a greeting|agent should make a square|agent should make a yellow circle|agent should answer circle|agent should answer with circle location|agent answer should make sense|agent should say they don't know how to do that",
    "agent should answer with their location|agent should say a capability|agent should answer with your location|agent should spawn two pigs|agent should answer with location of a pig|agent should go to one of the pigs|agent should follow the sheep if and only if there are any",
    "agent should dig two holes|agent should fill one hole|agent should make two circles|agent should face the circle|agent should say there is no object marked as favorite|agent should go to the circle|agent should follow you",
    "agent should move to you|agent should move to where you are looking|agent should turn left|agent should turn right|agent should move forward one step|agent should walk backward|agent should move to your left|agent should move to your right",
]
