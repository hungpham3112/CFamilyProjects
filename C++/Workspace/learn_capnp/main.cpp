#include "person.capnp.h"
#include <capnp/serialize.h>
#include <fcntl.h> // open()
#include <iostream>
#include <kj/debug.h>
#include <kj/io.h>
#include <unistd.h> // close()

int main()
{
    // BUILDING MESSAGE
    capnp::MallocMessageBuilder message;
    Person::Builder person = message.initRoot<Person>();
    person.setId(123);
    person.setName("Alice");

    // WRITE to file using KJ low-level API
    int fd = open("person.bin", O_CREAT | O_WRONLY | O_TRUNC, 0666);
    KJ_REQUIRE(fd >= 0, "Failed to open file for writing");
    kj::FdOutputStream outputStream(fd);
    capnp::writeMessage(outputStream, message);
    close(fd);

    // READ from file using KJ
    fd = open("person.bin", O_RDONLY);
    KJ_REQUIRE(fd >= 0, "Failed to open file for reading");
    kj::FdInputStream inputStream(fd);
    capnp::InputStreamMessageReader reader(inputStream);
    Person::Reader readPerson = reader.getRoot<Person>();

    std::cout << "ID: " << readPerson.getId() << std::endl;
    std::cout << "Name: " << readPerson.getName().cStr() << std::endl;

    close(fd);
    return 0;
}
