#ifndef __C_UTIL_STOPWATCH_HPP__
#define __C_UTIL_STOPWATCH_HPP__

#include <iostream>
#include <cstdio>

#include "c_util.hpp"

// includes, system
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#else
#include <ctime>
#include <sys/time.h>
#endif

/*********************************************************/
/* @class name StopWatch                                 */
/* @brief Time measurement utility for windows and linux */
/*********************************************************/
class StopWatch {

public:

    /* Constructor and destructor */
    inline StopWatch();
    inline ~ StopWatch();

    /* Start time measurement */
    inline void start();

    /* Stop time measurement */
    inline void stop();

    /* Reset time counters to zero */
    inline void reset();

    /* Print the total execution time  */
    inline int print_total_time(const char *str);

    /* Print the diff execution time  */
    inline int print_diff_time(const char *str);

    /* Print the average execution time  */
    inline int print_average_time(const char *str);

    /* Time in msec. after start. If the stop watch is still running (i.e. there */
    /* was no call to stop()) then the elapsed time is returned, otherwise the */
    /* time between the last start() and stop call is returned */
    inline float get_time() const;

    inline float get_diff_time() const;

    /* Mean time to date based on the number of times the stopwatch has been */
    /* _stopped_ (ie finished sessions) and the current total time */
    inline float get_average_time() const;

    inline StopWatch & operator=(const float tm);
    inline StopWatch & operator+=(const float tm);

private:

    /* member variables */

    /* Start of measurement */
#ifdef _WIN32
    LARGE_INTEGER start_time;
    LARGE_INTEGER end_time;
#else
    struct timeval start_time;
    struct timeval t_time;
#endif

    /* Time difference between the last start and stop */
    float diff_time;

    /* TOTAL time difference between starts and stops */
    float total_time;

    /* flag if the stop watch is running */
    bool running;

    /* Number of times clock has been started */
    /* and stopped to allow averaging */
    int clock_sessions;

#ifdef _WIN32
    double freq;
    bool freq_set;
#endif
};

StopWatch::StopWatch()
{
    diff_time = 0;
    total_time = 0;
    clock_sessions = 0;
#ifdef _WIN32
    freq_set = false;
    if (!freq_set) {
        LARGE_INTEGER temp;
        QueryPerformanceFrequency((LARGE_INTEGER *) & temp);
        freq = ((double) temp.QuadPart) / 1000.0;
        freq_set = true;
    }
#endif
}

StopWatch::~StopWatch()
{
}

inline void
StopWatch::start()
{
#ifdef _WIN32
    QueryPerformanceCounter((LARGE_INTEGER *) & start_time);
#else
    gettimeofday(&start_time, 0);
#endif
    running = true;
}

inline void StopWatch::stop()
{

#ifdef _WIN32
    QueryPerformanceCounter((LARGE_INTEGER *) & end_time);
    diff_time =
        (float) (((double) end_time.QuadPart -
                  (double) start_time.QuadPart) / freq);
#else
    gettimeofday(&t_time, 0);
    diff_time = (float) (1000.0 * (t_time.tv_sec - start_time.tv_sec)
                         +
                         (0.001 * (t_time.tv_usec - start_time.tv_usec)));
#endif
    total_time += diff_time;
    running = false;
    clock_sessions++;
}

inline void StopWatch::reset()
{
    diff_time = 0;
    total_time = 0;
    clock_sessions = 0;

    if (running) {
#ifdef _WIN32
        QueryPerformanceCounter((LARGE_INTEGER *) & start_time);
#else
        gettimeofday(&start_time, 0);
#endif
    }
}

inline int StopWatch::print_total_time(const char *str)
{
    fprintf(stderr, "Time ( %s ) = %.2lf msec\n", str, get_time());
    return 1;
}

inline int StopWatch::print_diff_time(const char *str)
{
    fprintf(stderr, "Diff Time ( %s ) = %.2lf msec\n", str, get_diff_time());
    return 1;
}

inline int StopWatch::print_average_time(const char *str)
{
    fprintf(stderr, "Avg Time of %d runs ( %s ) = %.2lf msec\n", clock_sessions, str,
            get_average_time());
    return 1;
}

inline float StopWatch::get_time() const
{
    // Return the TOTAL time to date
    float retval = total_time;

    if (running) {
        retval += get_diff_time();
    }

    return retval;
}

inline float StopWatch::get_diff_time() const
{
#ifdef _WIN32
    LARGE_INTEGER temp;
    QueryPerformanceCounter((LARGE_INTEGER *) & temp);
    return (float) (((double) (temp.QuadPart - start_time.QuadPart)) /
                    freq);
#else
    struct timeval t_time;
    gettimeofday(&t_time, 0);
    return (float) (1000.0 * (t_time.tv_sec - start_time.tv_sec)
                    + (0.001 * (t_time.tv_usec - start_time.tv_usec)));
#endif
}

inline float StopWatch::get_average_time() const
{
    return (clock_sessions > 0) ? (total_time / clock_sessions) : 0.0f;
}

inline StopWatch & StopWatch::operator=(const float tm)
{
    total_time = tm;
    diff_time = tm;
    clock_sessions = 1;
    return *this;
}

inline StopWatch & StopWatch::operator+=(const float tm)
{
    diff_time = tm;
    total_time += diff_time;
    clock_sessions++;
    return *this;
}

#endif                          // __C_UTIL_STOPWATCH_H_
