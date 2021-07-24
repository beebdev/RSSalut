# Credits: https://www.hackster.io/thatiotguy/enable-monitor-mode-in-tp-link-tl-wn722n-v2-v3-128fc6
# Tested on Kali Linux 2020.3
# Commands used to Setup the Adapter from aircrack repository
sudo apt update
sudo apt install bc -y
sudo rmmod r8188eu.ko
git clone https://github.com/aircrack-ng/rtl8188eus
cd rtl8188eus
sudo -i
echo "blacklist r8188eu" > "/etc/modprobe.d/realtek.conf"
exit
make
sudo make install
sudo modprobe 8188eu