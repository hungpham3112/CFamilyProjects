#include "exercise.h"
#include "munit.h"
#include <string.h>

munit_case(RUN, test_packet_header_size,
           { munit_assert_size(sizeof(packet_header_t), ==, 8, "PacketHeader union should be 8 bytes"); });

munit_case(RUN, test_tcp_header_fields, {
    packet_header_t header;
    header.tcp_header.src_port = 0x1234;
    header.tcp_header.dest_port = 0x5678;
    header.tcp_header.seq_num = 0x9ABCDEF0;

    munit_assert_uint16(header.tcp_header.src_port, ==, 0x1234, "src_port should be 0x1234");
    munit_assert_uint16(header.tcp_header.dest_port, ==, 0x5678, "dest_port should be 0x5678");
    munit_assert_uint32(header.tcp_header.seq_num, ==, 0x9ABCDEF0, "seq_num should be 0x9ABCDEF0");
});

munit_case(SUBMIT, test_field_raw_size,
           { munit_assert_size(sizeof(((packet_header_t *)0)->raw), ==, 8, "PacketHeader union should be 8 bytes"); });

munit_case(SUBMIT, test_field_to_raw_consistency, {
    packet_header_t header = {0};
    header.tcp_header.src_port = 0x1234;
    header.tcp_header.dest_port = 0x5678;
    header.tcp_header.seq_num = 0x9ABCDEF0;

    munit_assert_uint8(header.raw[0], ==, 0x34, "[0] should be 0x34");
    munit_assert_uint8(header.raw[1], ==, 0x12, "[1] should be 0x12");
    munit_assert_uint8(header.raw[2], ==, 0x78, "[2] should be 0x78");
    munit_assert_uint8(header.raw[3], ==, 0x56, "[3] should be 0x56");
    munit_assert_uint8(header.raw[4], ==, 0xF0, "[4] should be 0xF0");
    munit_assert_uint8(header.raw[5], ==, 0xDE, "[5] should be 0xDE");
    munit_assert_uint8(header.raw[6], ==, 0xBC, "[6] should be 0xBC");
    munit_assert_uint8(header.raw[7], ==, 0x9A, "[7] should be 0x9A");
});

int main()
{
    MunitTest tests[] = {
        munit_test("/test_packet_header_size", test_packet_header_size),
        munit_test("/test_tcp_header_fields", test_tcp_header_fields),
        munit_test("/test_field_raw_size", test_field_raw_size),
        munit_test("/test_field_to_raw_consistency", test_field_to_raw_consistency),
        munit_null_test,
    };

    MunitSuite suite = munit_suite("PacketHeader", tests);

    return munit_suite_main(&suite, NULL, 0, NULL);
}
