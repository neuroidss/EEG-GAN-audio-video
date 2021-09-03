#mkfifo messages
#socat PIPE:messages
socat -d -d -d -lpA_udp_rxr -u udp-recv:1111 - 2>>/dev/null |
socat -d -d -d -lpB_serial_adapter - /dev/ttyACM0,raw 2>>/dev/null |
socat -d -d -d -lpC_tcp_adapter - tcp4-listen:3333 2>>/dev/null |
socat -d -d -d -lpD_udp_sender -u - udp:localhost:1111,sourceport=2222 2>>/dev/null 
