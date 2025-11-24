#include <cstdint>
#include <iostream>
#include <string>
#include <thread> // Cần cho thread
#include <unistd.h>
#include <zmq.hpp>

void monitor_thread(zmq::context_t &context)
{
    // Tạo một socket đặc biệt để nhận các sự kiện monitor
    zmq::socket_t monitor(context, ZMQ_PAIR);

    // Kết nối tới địa chỉ "inproc" (in-process) mà publisher đã tạo
    try
    {
        monitor.connect("inproc://monitor.pub");
        std::cout << "[Monitor] Đang lắng nghe sự kiện...\n" << std::endl;

        while (true)
        {
            // Chờ nhận sự kiện
            zmq::message_t event_msg;
            std::optional<size_t> res = monitor.recv(event_msg);

            if (!res.has_value())
            {
                std::cerr << "[Monitor] Lỗi recv! Thoát thread monitor." << std::endl;
                break;
            }

            const auto *event_data = event_msg.data<zmq_event_t>();

            if (event_data->event == ZMQ_EVENT_ACCEPTED)
            {
                std::cout << ">>> [Monitor] SỰ KIỆN: Một Subscriber đã KẾT NỐI! <<<" << std::endl;
            }
            else if (event_data->event == ZMQ_EVENT_DISCONNECTED)
            {
                std::cout << ">>> [Monitor] SỰ KIỆN: Một Subscriber đã NGẮT KẾT NỐI! <<<" << std::endl;
            }

            // Nhận phần thứ 2 của tin nhắn (địa chỉ) và bỏ qua nó
            zmq::message_t addr_msg;
            auto monitor_res = monitor.recv(addr_msg);

            if (!monitor_res.has_value())
            {
                break;
            }
        }
    }
    catch (const zmq::error_t &e)
    {
        // Bắt lỗi nếu context bị hủy
        std::cerr << "[Monitor] Thread monitor bị ngắt." << std::endl;
    }
}

int main()
{

    zmq::context_t context(1);

    zmq::socket_t publisher(context, ZMQ_PUB);

    // Yêu cầu ZMQ gửi TẤT CẢ sự kiện tới địa chỉ "inproc" này
    int rc = zmq_socket_monitor(publisher, "inproc://monitor.pub", ZMQ_EVENT_ALL);
    if (rc != 0)
    {
        std::cerr << "Lỗi khi tạo monitor!" << std::endl;
        return 1;
    }

    std::thread t(monitor_thread, std::ref(context));
    t.detach(); // Cho nó chạy nền

    publisher.bind("tcp://*:5555");

    uint8_t count = 0;
    while (true)
    {
        std::string topic = "can";
        std::string data = "Hello" + std::to_string(count);

        zmq::message_t topic_msg(topic);
        zmq::message_t data_msg(data);

        try
        {
            publisher.send(topic_msg, zmq::send_flags::sndmore);
            publisher.send(data_msg, zmq::send_flags::none);
        }
        catch (const zmq::error_t &e)
        {
            std::cerr << "[Publisher] Lỗi send!" << std::endl;
            break;
        }

        std::cout << "[Pub] topic: " << topic << " data: " << data << std::endl;

        ++count;
        sleep(1);
    }
    return 0;
}
