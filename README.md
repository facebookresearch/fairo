<img style="float: right;" src="https://drive.google.com/uc?export=view&id=11tx9ZoQ9bP8SryqITN7wBP5cKOmMtS2I" width="300"/> </br>

**This repository, corresponding tutorials and docs are still being refined (and not ready yet).**

`droidlet` helps you rapidly build agents (real or virtual) that perform a wide variety of tasks specified by humans. The agents can use natural language, memory and humans in the loop.

`droidlet` is an ***early research project for AI researchers*** to explore ideas around *grounded dialogue*, *interactive learning* and *human-computer interfaces*.

`droidlet` is in active development and is fairly unstable in design, API, performance and correctness. It is not meant for any production use.

*Reach out to us at droidlet@fb.com, to discuss your use case or just share your thoughts!*

<p align="center">
   <img src="https://locobot-bucket.s3-us-west-2.amazonaws.com/documentation/droidlet.gif" />
</p>

## Getting Started

You want to do one of three things:

1. **Robots:** Reproduce and extend the [PyRobot](https://pyrobot.org) based agent on real robots such as [LocoBot](http://www.locobot.org/) or photo-realistic simulators such as [AIHabitat](https://aihabitat.org/).
2. **Minecraft:** Reproduced and extend the `minecraft` based game agent
3. **New Agent:** write your own agent from scratch, starting from our `base_agent` abstraction


<p align="center">
  <table align="center">
    <thead><th>Robots</th>
        <th>Minecraft</th>
        <th>New Agent</th>
    </thead>
    <tr valign="top">
        <td colspan="3"  align="left">
          1. Clone the source code
            <sub><pre lang="bash">
git clone --recursive https://github.com/facebookresearch/droidlet.git
cd droidlet
            </pre></sub>
        </td>    
    </tr>
    <tr valign="top">        
        <td> 2. Check system requirements
        <sub><pre lang="bash">
- Linux
- Python 3 (Anaconda recommended)
- NVIDIA GPU (8GB+)
- PyRobot-compatible robot or sim
  - Habitat-sim instructions below
        </pre></sub></td>
        <td><sub><pre lang="bash">
        <br/>
- Linux
- Python 3 (Anaconda recommended)
- NVIDIA GPU (4GB+)
- Minecraft
  - more instructions below
        </pre></sub></td>
        <td><sub><pre lang="bash">
        <br/>
- Linux
- Python 3 (Anaconda recommended)
        </pre></sub></td>
    </tr>
    <tr valign="top">        
        <td> 3. Install dependencies
        <sub><pre lang="bash">
conda create -n droidlet_env python=3.7 \
   pytorch==1.7.1 torchvision==0.8.2 \
   cudatoolkit=11.0 -c pytorch
conda activate droidlet_env
pip install -r \
    agents/locobot/requirements.txt
python setup.py develop
        </pre></sub></td>
        <td><sub><pre lang="bash">
        <br/>
pip install -r \
    agents/craftassist/requirements.txt
        </pre></sub></td>
        <td><sub><pre lang="bash">
        <br/>
pip install -r requirements.txt
        </pre></sub></td>
    </tr>
    <tr valign="top">        
        <td> 4. <a href="https://github.com/facebookresearch/droidlet/blob/main/agents/locobot/README.md"> Instructions for running the Locobot agent</a>
        </td>
        <td>
        <a href='https://github.com/facebookresearch/droidlet/blob/main/agents/craftassist/README.md'> Instructions for running the Craftassist agent</a>
        </td>
        <td>
        <br/>
        </td>
    </tr>
        <tr valign="top">
        <td colspan=3> 5. <a href="https://github.com/facebookresearch/droidlet/blob/main/tutorials"> Tutorials, runnable in Google Colab (more coming soon).</a><p> The tutorials introduce the `base_agent` architecture and take you through the 4 components of an Agent</p>
        </td>      
    </tr>    
    <tr valign="top" align="center">
        <td colspan=3> 6. <a href="https://facebookresearch.github.io/droidlet/"> API Documentation</a>
        </td>
    </tr>
    <tr valign="top" align="center">
        <td colspan=3> 7. Agent-specific API Documentation</a>
        </td>
    </tr>
    <tr valign="top">        
        <td align="center"><br/><a href="https://facebookresearch.github.io/droidlet/droidlet_agents.html#locobot"> Locobot agent API</a>
        </td>
        <td align="center">
        <br/><a href="https://facebookresearch.github.io/droidlet/droidlet_agents.html#craftassist"> CraftAssist agent API</a>
        </td>
        <td align="center">
        <br/>
        Not Applicable
        </td>
    </tr>
  </table>


## Documentation, Tutorials and Papers


Two papers cover the design of droidlet:
1. [droidlet: modular, heterogenous, multi-modal agents](https://arxiv.org/abs/2101.10384) covers the overall design of `droidlet` as an embodied AI platform that is extensible to physical robots and simulators.
2. [CraftAssist: A Framework for Dialogue-enabled Interactive Agents](https://arxiv.org/abs/1907.08584) covers the design of the dialogue parser and the task system of an earlier version of `droidlet` that is specific to the game [Minecraft](https://www.minecraft.net/en-us)

## Citation

If you use droidlet in your work, please cite our [arXiv paper](https://arxiv.org/abs/2101.10384):

```
@misc{pratik2021droidlet,
      title={droidlet: modular, heterogenous, multi-modal agents}, 
      author={Anurag Pratik and Soumith Chintala and Kavya Srinet and Dhiraj Gandhi and Rebecca Qian and Yuxuan Sun and Ryan Drew and Sara Elkafrawy and Anoushka Tiwari and Tucker Hart and Mary Williamson and Abhinav Gupta and Arthur Szlam},
      year={2021},
      eprint={2101.10384},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## License

droidlet is [MIT licensed](./LICENSE).


## Other Links

### Datasets

Download links to the datasets described in section 6 of [Technical Whitepaper](https://arxiv.org/abs/1907.08584) are provided here:

- **The house dataset**: https://craftassist.s3-us-west-2.amazonaws.com/pubr/house_data.tar.gz
- **The segmentation dataset**: https://craftassist.s3-us-west-2.amazonaws.com/pubr/instance_segmentation_data.tar.gz
- **The dialogue dataset**: https://craftassist.s3-us-west-2.amazonaws.com/pubr/dialogue_data.tar.gz

In the root of each tarball is a README that details the file structure contained within.



