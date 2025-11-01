#include <capnp/serialize.h>
#include <fcntl.h>  // for open()
#include <unistd.h> // for close()

int main(int argc, char **argv)
{
    int fd = open("addressbook.bin", O_RDONLY);
    ::capnp::StreamFdMessageReader message(fd);
    Person::Reader reader = message.getRoot<Person>();
    return 0;
}
