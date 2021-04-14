```eval_rst
.. _controller_label:
```
# Controller

In the "abstract" droidlet agent, the controller chooses whether to put Tasks on the Task Stack based on the memory state.   In the locobot agent and the craftassist agent subclasses, it consists of

* a [DSL](https://github.com/fairinternal/minecraft/blob/master/base_agent/documents/Action_Dictionary_Spec.md)
* a neural semantic parser, which translates natural language into partially specified programs over the DSL
* a Dialogue Manager, Dialogue Stack, and Dialogue objects.
* the Intepreter, a special Dialogue Object that takes partially specified programs from the DSL and fully specifies them using the Memory
* a set of "default behaviors", run randomly when the Task Stack and Dialogue Stack are empty
```eval_rst
Dialogue Objects behave similarly to :ref:`tasks_label` , except they only affect the agent's environment directly by causing the agent to issue utterances (or indirectly by pushing Task Objects onto the Task Stack).  In particular, each Dialogue Object has a .step() that is run when it is the highest priority object on the Stack. Dialogue Objects, like Task Objects are modular:  a learned model or a heuristic can mediate the Dialogue Object, and the same model or heuristic script can be used across many different agents.
```

The Dialogue Manager puts Dialogue Objects on the Dialogue Stack, either on its own, or at the request of a Dialogue Object.  In the locobot and craftassist agent, the manager is powered by a [neural semantic parser](https://github.com/fairinternal/minecraft/blob/master/base_agent/ttad/).

A sketch of the controller's operation is then
```
if new utterance from human:
     logical_form = semantic_parser.translate(new command)
     if the logical_form denotes a command:
         push Interpreter(logical_form, agent_memory) onto the DialogueStack
     else if the logical_form denotes some other kind of dialogue the agent can handle:
         push some other appropriate DialogueObject on the DialogueStack
if the Dialogue Stack is not empty:
     step the highest priority DialogueObject
if TaskStack is empty:
     maybe place default behaviors on the stack
```


## Dialogue Stack and Manager ##
The Dialogue Stack holds Dialogue Objects, and steps them.
```eval_rst
 .. autoclass:: base_agent.dialogue_stack.DialogueStack
  :members: peek, clear, append, step
```
The Dialogue Manager operates the Stack, and chooses whether to place Dialogue objects
```eval_rst
 .. autoclass:: base_agent.dialogue_manager.DialogueManager
  :members: step
```
### Semantic Parser ###
The training of the semantic parsing model we use is described in detail [here](https://github.com/fairinternal/minecraft/tree/master/base_agent/ttad/); the interface is
```eval_rst
 .. autoclass:: base_agent.ttad.ttad_transformer_model.query_model.TTADBertModel
  :members: parse
```
## Dialogue Objects ##
The generic Dialogue Object is
```eval_rst
 .. autoclass:: base_agent.dialogue_objects.dialogue_object.DialogueObject
   :members: step, check_finished
```
A DialogueObject's main method is .step(),
Some others:

```eval_rst
 .. autoclass:: base_agent.dialogue_objects.dialogue_object.Say
 .. autoclass:: base_agent.dialogue_objects.dialogue_object.AwaitResponse
 .. autoclass:: base_agent.dialogue_objects.dialogue_object.BotStackStatus
 .. autoclass:: base_agent.dialogue_objects.dialogue_object.ConfirmTask
 .. autoclass:: base_agent.dialogue_objects.dialogue_object.ConfirmReferenceObject

```



### Interpreter ###
The [Interpreter](https://github.com/fairinternal/minecraft/blob/master/base_agent/dialogue_objects/intepreter.py) is responsible for using the world state \(via [memory](memory.md)\) and a natural language utterance that has been parsed into a logical form over the agent's DSL from the semantic parser to choose a [Task](memory.md) to put on the Task Stack.   The [locobot](https://github.com/fairinternal/minecraft/blob/master/locobot/agent/dialogue_objects/loco_intepreter.py) and [craftassist](https://github.com/fairinternal/minecraft/blob/master/craftassist/agent/dialogue_objects/mc_intepreter.py) Interpreters are not the same, but the bulk of the work is done by the shared subinterpreters (in the files \*\_helper.py) [here](https://github.com/fairinternal/minecraft/blob/master/base_agent/dialogue_objects/).  The subinterpreters, registered in the main Interpreter [here](https://github.com/fairinternal/minecraft/blob/master/base_agent/dialogue_objects/intepreter.py#L55) \(and for the specialized versions [here](https://github.com/fairinternal/minecraft/blob/master/locobot/agent/dialogue_objects/loco_intepreter.py#L56) and [here](https://github.com/fairinternal/minecraft/blob/master/craftassist/agent/dialogue_objects/mc_intepreter.py#L61)\), roughly follow the structure of the DSL.  This arrangement is to allow replacing the (currently heuristic) subinterpreters with learned versions or specializing them to new agents.

```eval_rst
 .. autoclass:: base_agent.dialogue_objects.interpreter.Interpreter
```
