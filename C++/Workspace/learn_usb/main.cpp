#include <iomanip> // để dùng std::setw, std::setfill
#include <iostream>
#include <libusb-1.0/libusb.h>

int main()
{
    // Initialization
    libusb_context *ctx = NULL;
    if (libusb_init(&ctx) != 0)
    {
        printf("Init failed");
    };
    printf("Context is created at %p\n", ctx);

    // List all devices
    libusb_device **list;

    ssize_t num_device = libusb_get_device_list(ctx, &list);

    for (int i = 0; i < num_device; ++i)
    {
        libusb_device *device = list[i];
        struct libusb_device_descriptor desc;

        libusb_device_handle *handle;
        int err = libusb_open(device, &handle);
        if (err)
        {
            printf("Cannot open device: %d\n", err); // prints the libusb error code
            return 1;
        }

        // Use human-readable descriptor

        libusb_get_device_descriptor(device, &desc);
        unsigned char manufacturer[256] = "";
        unsigned char product[256] = "";
        uint8_t bus = libusb_get_bus_number(device);
        uint8_t addr = libusb_get_device_address(device);
        std::cout << "VID = 0x" << std::hex << desc.idVendor << std::endl;

        std::cout << "PID = 0x" << std::hex << desc.idProduct << std::endl;

        libusb_get_string_descriptor_ascii(handle, desc.iManufacturer, manufacturer, sizeof(manufacturer));
        libusb_get_string_descriptor_ascii(handle, desc.iProduct, product, sizeof(product));

        printf("Bus %03d Device %03d: ID %04x:%04x %s %s\n", bus, addr, desc.idVendor, desc.idProduct, manufacturer,
               product);
        libusb_close(handle);
    }

    printf("Number of devices: %ld\n", num_device);

    libusb_free_device_list(list, 1);
    // De-Initialization
    libusb_exit(ctx);
    return 0;
}
