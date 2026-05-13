#define _DEFAULT_SOURCE // enables struct ifreq
#include <linux/can.h>
#include <linux/sockios.h>
#include <net/if.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

int main()
{
    int fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    struct ifreq request;
    strncpy(request.ifr_name, "can0", IFNAMSIZ);
    ioctl(fd, SIOCGIFINDEX, &request);
    int idx = request.ifr_ifindex;

    struct sockaddr_can addr = {.can_family = PF_CAN, .can_ifindex = idx};
    bind(fd, (struct sockaddr *)&addr, sizeof(addr));

    struct can_frame frame = {
        .can_id = 123,
        .len = 8,
        .data = {0xDE, 0xAD, 0xBE, 0xEF},
    };
    write(fd, &frame, sizeof(frame));
    return 0;
}
