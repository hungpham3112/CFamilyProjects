#include <atomic>
#include <csignal>
#include <iostream>
#include <zmq.h>
#include <zmq.hpp>

std::atomic_bool running{true};

void signal_handler(int signal_num)
{
    if (signal_num == SIGINT || signal_num == SIGTERM)
    {
        std::cout << "\n>>> [Sub Broken] Đã bắt được SIGINT (Ctrl+C)! Đang set running = false; <<<" << std::endl;
        running = false;
    }
}
int main()
{
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    zmq::context_t context(1);

    zmq::socket_t subscriber(context, ZMQ_SUB);

    subscriber.connect("tcp://10.113.185.54:5555");

    const std::string topic = "can";
    subscriber.set(zmq::sockopt::subscribe, topic);

    std::cout << "Subscribed topic: " << topic << std::endl;
    while (running)
    {
        zmq::message_t topic_msg;
        zmq::message_t data_msg;
        try
        {

            std::optional<size_t> topic_res = subscriber.recv(topic_msg);

            if (!topic_res.has_value())
            {
                std::cerr << "There is no value in topic!" << std::endl;
                continue;
            }

            if (!topic_msg.more())
            {
                std::cerr << "No data in message!" << std::endl;
                continue;
            }

            std::optional<size_t> data_res = subscriber.recv(data_msg);
            if (!topic_res.has_value())
            {
                std::cerr << "Error when receiving data!" << std::endl;
                continue;
            }

            std::string topic = topic_msg.to_string();
            std::string data = data_msg.to_string();

            std::cout << "[Sub] topic: " << topic << " data: " << data << std::endl;
        }
        catch (const zmq::error_t &e)
        {
            // Đây là phần "Graceful" (Lịch sự)
            std::cout << ">>> ZMQ exception: " << e.what() << std::endl;

            // Kiểm tra xem có phải lỗi "Interrupted system call" không
            if (e.num() == EINTR)
            {
                // Đây là lỗi mong đợi khi nhấn Ctrl+C
                std::cout << ">>> Bị ngắt (Interrupted). Kiểm tra cờ 'running'..." << std::endl;
                // Không làm gì cả, vòng lặp while(running) sẽ tự động
                // kiểm tra cờ 'running' ở lần lặp tiếp theo và thoát ra.
                continue;
            }
            else
            {
                // Đây là một lỗi ZMQ thực sự khác
                std::cerr << "!!! Lỗi ZMQ không mong đợi. Đang thoát..." << std::endl;
                running = false; // Thoát vòng lặp
            }
        }
    }
    std::cout << "[Sub Broken] Vòng lặp đã thoát. Đang dọn dẹp (RAII)..." << std::endl;
    return 0;
}
