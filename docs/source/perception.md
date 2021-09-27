```eval_rst
.. _perception_label:
```
# Perception
Any perceptual information to be used by the agent controllers should be stored in [Memory](memory.md), and accessed through the database methods, memory nodes or the memory-query subset of the DSL.  This arrangement makes it easier to swap out perceptual modules and add new ones, and allows that agents with different underlying hardware and perceptual capabilities can share other aspects of the agent.  We also hope this will give hooks for representation learning and especially self-supervised representation learning.


Perceptual modules are registered with an agent in the agent's perceptual_modules dict, see e.g. [here](https://github.com/facebookresearch/fairo/blob/main/agents/locobot/locobot_agent.py#L206) and
[here](https://github.com/facebookresearch/fairo/blob/main/agents/craftassist/craftassist_agent.py#L186).  Perceptual modules have one required method, .perceive(), which should add or update information in the Memory as appropriate.  This method is called on each iteration of the agent's main loop [here](https://github.com/facebookresearch/fairo/blob/main/agents/loco_mc_agent.py#L281).  A perceptual module which is slow might run only once in a while (keeping its own time; the agent will call it each step).  If some part of the agent really needs a .perceive(), it can signal this by calling with force=True.
