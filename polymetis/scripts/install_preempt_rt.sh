# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Exit on the first error.
set -e

VERSION=4.14.78
VERSION_PATCH=4.14.78-rt47
DEFAULT_CONFIG=/boot/config-4.15.0-51-generic

if [  ! -f  $DEFAULT_CONFIG ]; then
   echo "Configure file $FILE does not exist. Please use other file."
   exit -1
fi

echo "========================================================================="
echo "==="
echo "=== Building kernel in ~/Downloads/rt_preempt_kernel_install"
echo "==="
echo "========================================================================="

# Install dependencies to build kernel.
sudo apt-get install -y libelf-dev libncurses5-dev libssl-dev kernel-package

# Install packages to test rt-preempt.
sudo apt install rt-tests

# Create folder to build kernel.
mkdir -p ~/Downloads/rt_preempt_kernel_install
cd ~/Downloads/rt_preempt_kernel_install

# Download kernel version and patches.
wget -nc https://mirrors.edge.kernel.org/pub/linux/kernel/v4.x/linux-$VERSION.tar.xz
wget -nc http://cdn.kernel.org/pub/linux/kernel/projects/rt/4.14/older/patch-$VERSION_PATCH.patch.xz
xz -cd linux-$VERSION.tar.xz | tar xvf -

# Apply patch
cd linux-$VERSION/
xzcat ../patch-$VERSION_PATCH.patch.xz | patch -p1

# Create necessary file, see: https://ubuntuforums.org/showthread.php?t=2373905
touch REPORTING-BUGS

# Copy default config and prompt for configuration screen.
cp $DEFAULT_CONFIG .config

echo "Please apply the following configurations in the next step:"
echo ""
echo "General setup"
echo "  Local version - append to kernel release: [Enter] Add '-preempt-rt'"
echo ""
echo "Processor type and features ---> [Enter]"
echo "  Preemption Model (Voluntary Kernel Preemption (Desktop)) [Enter]"
echo "    Fully Preemptible Kernel (RT) [Enter] #Select"
echo ""

read -p "Please read the above instructions" yn

make menuconfig -j


# Build the kernel.
NUMBER_CPUS=`grep -c ^processor /proc/cpuinfo`
CONCURRENCY_LEVEL=$NUMBER_CPUS make-kpkg --rootcmd fakeroot --initrd kernel_image kernel_headers

# Install the build kernel.
sudo dpkg -i ../linux-headers-$VERSION_PATCH-preempt-rt_$VERSION_PATCH-preempt-rt-10.00.Custom_amd64.deb ../linux-image-$VERSION_PATCH-preempt-rt_$VERSION_PATCH-preempt-rt-10.00.Custom_amd64.deb


# Modify the grub setting: comment out GRUB_HIDDEN_TIMEOUT and update grub.
sudo sed -i 's/GRUB_HIDDEN_TIMEOUT/# GRUB_HIDDEN_TIMEOUT/g' /etc/default/grub
sudo update-grub

# Create realtime config.
if [  ! -f  /etc/security/limits.d/99-realtime.conf ]; then
  sudo tee /etc/security/limits.d/99-realtime.conf > /dev/null <<EOL
@realtime   -   rtprio  99
@realtime   -   memlock unlimited
EOL
fi


if grep -q "realtime" /etc/group; then
  echo "Realtime group already exists"
else
  sudo groupadd realtime
fi

sudo usermod -a -G realtime $USER

# Change the permission on /dev/cpu_dma_latency. This allows other users to
# set the minimum desired latency for the CPU other than root (e.g. the current
# user from dynamic graph manager).
sudo chmod 0777 /dev/cpu_dma_latency

echo "========================================================================="
echo "==="
echo "=== Installation done. Please reboot and select new kernel from grub menu."
echo "==="
echo "=== Make sure to add all uses with rt permissions to the 'realtime' group using:"
echo "==="
echo "===  sudo usermod -a -G realtime $USER"
echo "==="
echo "========================================================================="

