#include <hikari/thread/spin_lock.h>
#include <thread>

#if defined(_MSC_VER) && ((_M_IX86_FP >= 2) || defined(_M_X64))
#    include <emmintrin.h>
#    define PAUSE _mm_pause
#elif (defined(__clang__) || defined(__GNUC__)) && (defined(__i386__) || defined(__x86_64__))
#    define PAUSE __builtin_ia32_pause
#elif (defined(__clang__) || defined(__GNUC__)) && (defined(__arm__) || defined(__aarch64__))
#    define PAUSE() asm volatile("yield")
#else
#    define PAUSE()
#endif

void hikari::SpinLock::wait()
{
  constexpr size_t attemp_to_finish_loop = 64;
  for (size_t i = 0; i < attemp_to_finish_loop; ++i) {
    if (!is_locked()) break;
    PAUSE();
  }
  std::this_thread::yield();
}
