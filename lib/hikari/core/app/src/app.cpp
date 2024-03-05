#include <hikari/core/app/app.h>

hikari::core::App:: App(int argc, const char* argv[]) noexcept {}
hikari::core::App::~App() noexcept {}
int hikari::core::App::run()
{
    bool is_success = true;
    try
    {
        if (initialize())
        {
            mainLoop();
        }
        else
        {
            is_success = false;
        }
    }
    catch (std::exception &e)
    {
        is_success = false;
    }
    terminate();
    return is_success ? 0 : -1;
}
