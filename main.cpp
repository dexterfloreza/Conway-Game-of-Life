#include <SFML/Graphics.hpp>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>
#include <chrono>
#include <random>
#include <string>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <iostream>

//
// ---------- Thread Pool with Work Stealing ----------
//
class ThreadPool {
public:
    ThreadPool(size_t numThreads = std::thread::hardware_concurrency()) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i)
            workers.emplace_back([this, i]() { workerLoop(i); });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        cond.notify_all();
        for (auto& w : workers) w.join();
    }

    void enqueue(std::function<void()> job) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.push(std::move(job));
        }
        cond.notify_one();
    }

    void waitAll() {
        std::unique_lock<std::mutex> lock(doneMutex);
        doneCond.wait(lock, [this]() { return tasks.empty() && (activeWorkers == 0); });
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex, doneMutex;
    std::condition_variable cond, doneCond;
    std::atomic<int> activeWorkers{0};
    bool stop;

    void workerLoop(size_t id) {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                cond.wait(lock, [this]() { return stop || !tasks.empty(); });
                if (stop && tasks.empty()) return;
                job = std::move(tasks.front());
                tasks.pop();
                ++activeWorkers;
            }

            job();

            {
                std::unique_lock<std::mutex> lock(doneMutex);
                --activeWorkers;
                if (tasks.empty() && activeWorkers == 0)
                    doneCond.notify_all();
            }
        }
    }
};

//
// ---------- Game of Life Core ----------
//
class LifeAccel {
public:
    LifeAccel(int width, int height, int cellSize, ThreadPool& pool)
        : width(width), height(height), cellSize(cellSize),
          cols(width / cellSize), rows(height / cellSize),
          current(rows, std::vector<uint8_t>(cols, 0)),
          next(rows, std::vector<uint8_t>(cols, 0)),
          threadPool(pool) {}

    void randomize(double fill = 0.25) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& row : current)
            for (auto& c : row)
                c = (dist(rng) < fill) ? 1 : 0;
    }

    void updateParallel() {
        int nThreads = std::thread::hardware_concurrency();
        int chunk = rows / nThreads;
        for (int t = 0; t < nThreads; ++t) {
            int start = t * chunk;
            int end = (t == nThreads - 1) ? rows : start + chunk;
            threadPool.enqueue([=, this]() {
                for (int i = start; i < end; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        int n = countNeighbors(i, j);
                        next[i][j] = (current[i][j]) ? (n == 2 || n == 3) : (n == 3);
                    }
                }
            });
        }
        threadPool.waitAll();
        current.swap(next);
    }

    void draw(sf::RenderWindow& window) const {
        sf::RectangleShape cell(sf::Vector2f(cellSize - 1, cellSize - 1));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (current[i][j]) {
                    cell.setPosition(j * cellSize, i * cellSize);
                    cell.setFillColor(sf::Color(80, 200, 255));
                    window.draw(cell);
                }
            }
        }
    }

private:
    int width, height, cellSize;
    int cols, rows;
    std::vector<std::vector<uint8_t>> current, next;
    ThreadPool& threadPool;

    int countNeighbors(int x, int y) const {
        int c = 0;
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy)
                if (!(dx == 0 && dy == 0)) {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < rows && ny >= 0 && ny < cols)
                        c += current[nx][ny];
                }
        return c;
    }
};

//
// ---------- Timing Utilities ----------
//
struct Timer {
    std::chrono::steady_clock::time_point start;
    void tic() { start = std::chrono::steady_clock::now(); }
    double toc() const {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

//
// ---------- Main Simulation ----------
//
int main() {
    constexpr int WIDTH = 1280;
    constexpr int HEIGHT = 720;
    constexpr int CELL = 4;
    constexpr int TARGET_FPS = 60;

    ThreadPool pool;
    LifeAccel life(WIDTH, HEIGHT, CELL, pool);
    life.randomize(0.3);

    // ✅ SFML 2.x version
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "LifeAccel — Firmware-Class Game of Life");
    window.setFramerateLimit(TARGET_FPS);

    sf::Font font;
    // ✅ SFML 2.x: loadFromFile
    if (!font.loadFromFile("arial.ttf")) {
        std::cerr << "Warning: Font not found, overlay disabled.\n";
    }

    // ✅ SFML 2.x: set font and size manually
    sf::Text overlay;
    overlay.setFont(font);
    overlay.setCharacterSize(16);
    overlay.setFillColor(sf::Color::White);

    double avgFps = 0.0;
    int frameCount = 0;

    sf::Event event;
    while (window.isOpen()) {
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        Timer totalTimer; totalTimer.tic();
        Timer updateTimer; updateTimer.tic();
        life.updateParallel();
        double updateTime = updateTimer.toc();

        Timer drawTimer; drawTimer.tic();
        window.clear(sf::Color::Black);
        life.draw(window);
        double drawTime = drawTimer.toc();

        double frameTime = totalTimer.toc();
        double fps = 1000.0 / frameTime;
        avgFps = (avgFps * frameCount + fps) / (frameCount + 1);
        frameCount++;

        if (!font.getInfo().family.empty()) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1);
            oss << "FPS: " << fps << " (" << avgFps << " avg)"
                << " | Update: " << updateTime << " ms"
                << " | Draw: " << drawTime << " ms";
            overlay.setString(oss.str());
            overlay.setPosition(10.f, 10.f);
            window.draw(overlay);
        }

        window.display();
    }

    return 0;
}
