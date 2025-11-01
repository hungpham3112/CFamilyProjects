#include "addressbook.capnp.h"
#include <capnp/serialize.h>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

int main()
{
    ::capnp::MallocMessageBuilder message;
    auto person = message.initRoot<Person>();
    person.setId(123);
    person.setName("Lz");

    // 0666 (octal)= 0b110 110 110 (binary) = r+w for host, group, other
    // O_ stands for "open and", O_WRONLY = Open and WRite ONLY, O_CREAT = Open and CREATe if it isn't created,
    // O_TRUNC = Open and TRUCation (Remove all before write)
    int fd = open("addressbook.bin", O_WRONLY | O_CREAT | O_TRUNC, 0666);
    ::capnp::writeMessageToFd(fd, message);
    close(fd);
}
