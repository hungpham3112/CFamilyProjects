#define _DEFAULT_SOURCE
#include <linux/can.h>
#include <linux/can/raw.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

int main(void)
{
    int fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (fd < 0) { perror("socket"); return 1; }

    struct ifreq ifr = {0};
    strncpy(ifr.ifr_name, "can0", IFNAMSIZ - 1);
    if (ioctl(fd, SIOCGIFINDEX, &ifr) < 0) { perror("ioctl"); return 1; }

    struct sockaddr_can addr = {
        .can_family  = AF_CAN,
        .can_ifindex = ifr.ifr_ifindex,
    };
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }

    struct can_frame frame = {
        .can_id = 0x123,
        .len    = 4,
        .data   = {0xDE, 0xAD, 0xBE, 0xEF},
    };
    if (write(fd, &frame, sizeof(frame)) < 0) { perror("write"); return 1; }

    puts("frame sent ok");
    return 0;
}
