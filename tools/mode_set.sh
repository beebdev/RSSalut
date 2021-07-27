#!/bin/bash

iface="wlxd037457e7948"
mon_mode=false

_usage="usage: $0 [options]
Description: Enables/disables monitor mode for specified network interface.

Options              Description
-h, --help           help
-i, --interface      Specify interface to set mode on. Default: ${iface}
-I, --monitor-mode   Enable monitor mode for interface
"

while test $# -gt 0; do
    case "$1" in
        -h|--help)
            echo "$_usage"
            exit 0
            ;;
        -i|--interface)
            shift
            echo "Interface: $1"
            iface=$1
            shift
            ;;
        -I|--monitor-mode)
            echo "Enable monitor mode."
            mon_mode=true
            shift
            ;;
        *)
            echo "$_usage"
            exit 0
            ;;
    esac
done

echo "dsdfsdfs"

if [ "$mon_mode" = true ]; then
    echo "Enabling monitor mode for $iface"
    airmon-ng start ${iface}
    iwconfig
else
    echo "Disabling monitor mode for $iface"
    airmon-ng stop ${iface}
    service network-manager start
    iwconfig
fi