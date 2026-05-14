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
if (fd < 0) { perror("socket"); return 1; }  // check fd FIRST

int disable = 0;
if (setsockopt(fd, SOL_CAN_RAW, CAN_RAW_LOOPBACK, &disable, sizeof(disable)) < 0)
{
    perror("setsockopt CAN_RAW_LOOPBACK");
    return 1;
}

    struct ifreq ifr;
    strncpy(ifr.ifr_name, "can0", IFNAMSIZ - 1);
    if (ioctl(fd, SIOCGIFINDEX, &ifr) < 0)
    {
        perror("ioctl");
        return 1;
    }

    struct sockaddr_can addr = {
        .can_family = AF_CAN,
        .can_ifindex = ifr.ifr_ifindex
    };

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {
        perror("bind");
        return 1;
    }

    while (1) {
    struct can_frame frame;
    if (read(fd, &frame, sizeof(frame)) < 0)
    {
        perror("write");
        return 1;
    }

    printf("%s %x [%d] ", ifr.ifr_name, frame.can_id, frame.len);
    for (int i = 0; i < frame.len; i++)
    {
        printf("%02x ", frame.data[i]);
    }
    printf("\n");
    }
    return 0;
}
