#mkfifo messages
#socat PIPE:messages
socat -d -d -d -lpA_udp_rxr -u udp-recv:1112 - 2>>/dev/null |
socat -d -d -d -lpB_serial_adapter - /dev/ttyUSB0,b921600,raw 2>>/dev/null |
socat -d -d -d -lpC_tcp_adapter - tcp4-listen:3334 2>>/dev/null |
socat -d -d -d -lpD_udp_sender -u - udp:localhost:1112,sourceport=2223 2>>/dev/null 
