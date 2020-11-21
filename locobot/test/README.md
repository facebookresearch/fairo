Before following these instructions, make sure you are allowed by penguin to run docker.  Sample task to get permission: T69617477

# Locobot Assistant Unit tests

There are two sets of tests in this folder:

1. Smoke tests
2. Unit tests


## Running Tests

An easy way to do this is to run it in a controlled docker environment with all relevant dependencies such as habitat. You can do so, for example with:

```bash
cd ../../../              # goes from "minecraft/python/locobot/test" to the root of the repo
docker run --gpus all  --ipc=host -v $(pwd):/remote -w /remote theh1ghwayman/locobot-assistant:5.0 .circleci/locobot_tests.sh
```

## Manually and iteratively running particular tests during development

Make sure that the locobot / habitat environment is running.
One way to do this is to run the script (relative to this folder):

Open a new Terminal and run:

```bash
cd ../../../              # goes from "minecraft/python/locobot/test" to "minecraft/", i.e the root of the repo
docker run --gpus all -it --rm --ipc=host -v $(pwd):/remote -w /remote theh1ghwayman/locobot-assistant:5.0 bash
roscore &
load_pyrobot_env
cd locobot/robot
./launch_pyro_habitat.sh
```

Now that the habitat environment is ready to connect, in your development terminal, run these whenever you want to retest after making code changes:

```bash
export LOCOBOT_IP="172.17.0.2" # docker's client IP
python test_habitat.py # unit tests
python smoke_test.py   # smoke tests
```

## Adding or modifying visual tests

When adding new visual tests (i.e. adding `assert_visual` calls), or when modifying existing tests wrt environment changes, the recorded images would either be missing or be outdated.

To write new "expected" images, you can run with the environment variable `EXPECTTEST_ACCEPT` enabled.

For example:

```bash
python test_habitat.py    # failed because of known environment or test changes

EXPECTTEST_ACCEPT=1 \
python test_habitat.py    # rewrites expected images

python test_habitat.py    # passes again
```


