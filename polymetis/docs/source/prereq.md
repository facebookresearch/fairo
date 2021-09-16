# Prerequisites

## Simulation environments only

For **simulation**, we simply assume a [Conda](https://docs.conda.io/en/latest/) environment with Python 3.8, running on a Linux machine. You may then proceed to [installation](installation.md).

## Franka Panda hardware setup

To control real-time-capable **hardware** (such as the Franka Panda), we highly recommend running the server on a separate machine (e.g. a [NUC](https://www.amazon.com/gp/product/B0842WKBCF/)) with a real-time kernel which is directly connected to the robot.
Your client code should then connect to the server over network and execute any heavy computation (e.g. image processing, planning, GPU usage) to avoid interrupting the real-time control loop on the NUC.

### Setting up the Franka Panda

1. [Mounting plate](https://vention.io/parts/franka-emika-panda-mounting-plate-487) and [clamps](https://www.amazon.com/gp/product/B001KPVFJE/)[^1]

1. Robot stand.
    1. A pricier but nice solution is this [custom-designed](https://vention.io/designs/98080) stand.

1. Networking
    1. Connect NUC to Control using Ethernet
    2. Set the "Wired" network settings on the NUC for control: Manual IPv4 with Address `172.16.0.1`, Netmask `255.255.255.0`
    3. Put GPU-enabled workstation on the same network (could be wireless, since no real-time guarantees for user code).
    4. Optional: You can use [FoxyProxy](https://addons.mozilla.org/en-US/firefox/addon/foxyproxy-standard/) to access Franka Desk from your user machine.
        1. In your `~/.ssh/config`, find the entry you use to `ssh` into your NUC. Add `DynamicForward 1337` to that entry and use that port
        1. Add this to your FoxyProxy settings: ![FoxyProxy](img/foxyproxy.png)

1. Robot firmware upgrade through Franka World. See Franzi's [Facebook-internal note](https://fb.workplace.com/notes/762408351366858) for details

### Setting up the NUC

_The following guide is written with specific, known-good version numbers to get you up and running as quickly as possible; other versions will also likely work._

1. Install [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)

1. The Franka documentation has a [comprehensive guide](https://frankaemika.github.io/docs/installation_linux.html#setting-up-the-real-time-kernel) on installing real-time kernel. Here is a condensed version[^2]:
    1. Install prereqs:

            sudo apt install build-essential bc curl ca-certificates gnupg2 libssl-dev lsb-release libelf-dev bison flex

    1. Download known-good kernel and path:
    
            curl -SLO https://mirrors.edge.kernel.org/pub/linux/kernel/v5.x/linux-5.11.tar.xz
            curl -SLO https://mirrors.edge.kernel.org/pub/linux/kernel/projects/rt/5.11/older/patch-5.11-rt7.patch.xz
            xz -d linux-5.11.tar.xz
            xz -d patch-5.11-rt7.patch.xz

    1. Extract the kernel and apply the patch:
            
            tar xf linux-5.11.tar
            cd linux-5.11
            patch -p1 < ../patch-5.11-rt7.patch
            
    1. Configure the kernel:
    
            make oldconfig
       
       Choose `Fully Preemptible Kernel` when asked for Preemption Model, and leave the rest to defaults (keep pressing `Enter`).
       
       Set the following values in the `.config` file:
        - `CONFIG_SYSTEM_TRUSTED_KEYS = ""`
        - `CONFIG_MODULE_SIG_KEY="certs/signing_key.pem"`
        - `CONFIG_SYSTEM_TRUSTED_KEYRING=y`

    3. Compile the kernel: `sudo make -j4 deb-pkg`
        - This takes a long time, so set `-j` to use more cores.
        - If you get `kernel signature invalid` error, disable secure boot in your BIOS.
    
    4. Install the kernel:
            
            sudo dpkg -i ../linux-headers-5.11.0-rt7_*.deb ../linux-image-5.11.0-rt7_*.deb
    
    5. Follow the rest of the instructions in [Franka's guide](https://frankaemika.github.io/docs/installation_linux.html#verifying-the-new-kernel) to verify the kernel and allow a user to set real-time permissions.

You are now ready to install Polymetis.

---

[^1]: Credit to Vikash Kumar for documenting the hardware installation process.

[^2]: Credit to Akshara Rai for going through the process and debugging these quirks.
