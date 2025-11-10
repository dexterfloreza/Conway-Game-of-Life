#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>
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
#include <cmath>

//
// ---------- Thread Pool ----------
//
class ThreadPool {
public:
    ThreadPool(size_t n = std::thread::hardware_concurrency()) : stop(false) {
        for (size_t i = 0; i < n; ++i)
            workers.emplace_back([this]() { workerLoop(); });
    }
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(qMutex);
            stop = true;
        }
        cond.notify_all();
        for (auto &t : workers) t.join();
    }
    void enqueue(std::function<void()> job) {
        {
            std::unique_lock<std::mutex> lock(qMutex);
            tasks.push(std::move(job));
        }
        cond.notify_one();
    }
    void waitAll() {
        std::unique_lock<std::mutex> lock(doneMutex);
        doneCond.wait(lock, [this]() { return tasks.empty() && activeWorkers == 0; });
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex qMutex, doneMutex;
    std::condition_variable cond, doneCond;
    std::atomic<int> activeWorkers{0};
    bool stop;

    void workerLoop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(qMutex);
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
// ---------- LifeAccel ----------
//
class LifeAccel {
public:
    LifeAccel(int w, int h, int c, ThreadPool &p)
        : width(w), height(h), cellSize(c),
          cols(w / c), rows(h / c),
          current(rows, std::vector<uint8_t>(cols, 0)),
          next(rows, std::vector<uint8_t>(cols, 0)),
          pool(p) {}

    void randomize(double fill = 0.25) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> dist(0, 1);
        for (auto &r : current)
            for (auto &c : r)
                c = dist(rng) < fill ? 1 : 0;
    }

    void updateParallel() {
        int nThreads = std::thread::hardware_concurrency();
        int chunk = rows / nThreads;
        for (int t = 0; t < nThreads; ++t) {
            int start = t * chunk;
            int end = (t == nThreads - 1) ? rows : start + chunk;
            pool.enqueue([=, this]() {
                for (int i = start; i < end; ++i)
                    for (int j = 0; j < cols; ++j) {
                        int n = countNeighbors(i, j);
                        next[i][j] = current[i][j] ? (n == 2 || n == 3) : (n == 3);
                    }
            });
        }
        pool.waitAll();
        current.swap(next);
    }

    void draw(sf::RenderWindow &win) const {
        sf::RectangleShape cell(sf::Vector2f(cellSize - 1, cellSize - 1));
        cell.setFillColor(sf::Color(80, 200, 255));
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                if (current[i][j]) {
                    cell.setPosition(j * cellSize, i * cellSize);
                    win.draw(cell);
                }
    }

    int getLiveCount() const {
        int c = 0;
        for (auto &r : current)
            for (auto v : r)
                c += v;
        return c;
    }

private:
    int width, height, cellSize, cols, rows;
    std::vector<std::vector<uint8_t>> current, next;
    ThreadPool &pool;

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
// ---------- Simulation Metrics ----------
//
struct SimulationMetrics {
    double fps = 0, avgFps = 0, updateMs = 0, frameMs = 0;
    int live = 0, delta = 0;
    long long gen = 0;
};

void updateMetricsWindow(sf::RenderWindow &win, const SimulationMetrics &m, sf::Font &font) {
    win.clear(sf::Color(20, 20, 30));
    sf::Text h("SIMULATION METRICS", font, 20);
    h.setFillColor(sf::Color(255, 200, 0));
    h.setPosition(20, 10);
    win.draw(h);

    std::ostringstream s;
    s << std::fixed << std::setprecision(1)
      << "FPS: " << m.fps << " (" << m.avgFps << " avg)\n"
      << "Update: " << m.updateMs << " ms\n"
      << "Frame: " << m.frameMs << " ms\n"
      << "Live Cells: " << m.live << "\n"
      << "Δ Cells: " << m.delta << "\n"
      << "Generation: " << m.gen;
    sf::Text body(s.str(), font, 16);
    body.setFillColor(sf::Color(180, 220, 255));
    body.setPosition(20, 50);
    win.draw(body);
    win.display();
}

//
// ---------- Pixel DNA Renderer ----------
//
void drawPixelDNA(sf::RenderWindow &window, sf::Vector2f pos, float scale, float time) {
    sf::Color blue(0, 170, 255);
    sf::Color green(90, 230, 120);
    sf::Color white(240, 240, 255);

    const int height = 30;
    const int segments = 32;
    const float wavelength = 10.0f;
    const float amplitude = 6.0f;

    // two smooth curves
    sf::VertexArray left(sf::LineStrip, segments);
    sf::VertexArray right(sf::LineStrip, segments);

    for (int i = 0; i < segments; ++i) {
        float x = i * wavelength * scale * 0.1f;
        float y = std::sin((i * 0.5f) + time * 2.f) * amplitude * scale;

        left[i].position = {pos.x + x, pos.y + y};
        right[i].position = {pos.x + x, pos.y - y};
        left[i].color = blue;
        right[i].color = green;
    }

    // horizontal rungs
    for (int i = 0; i < segments; i += 2) {
        sf::Vertex line[] = {
            sf::Vertex(left[i].position, white),
            sf::Vertex(right[i].position, white)
        };
        window.draw(line, 2, sf::Lines);
    }

    window.draw(left);
    window.draw(right);
}


//
// ---------- Title Screen ----------
//
void showTitleScreen(sf::RenderWindow &win) {
    sf::Music mus;
    if (mus.openFromFile("title_conway.mp3")) {
        mus.setLoop(true);
        mus.setVolume(60);
        mus.play();
    }

    sf::Font f;
    if (!f.loadFromFile("VCR_OSD_MONO_1.001.ttf")) {
        std::cerr << "Font not found!\n";
        return;
    }

    sf::Text title("CONWAY'S\nGAME OF LIFE", f, 64);
    title.setFillColor(sf::Color(255, 230, 0));
    title.setStyle(sf::Text::Bold);
    title.setLetterSpacing(2);
    title.setPosition(
        (win.getSize().x - title.getGlobalBounds().width) / 2,
        (win.getSize().y - title.getGlobalBounds().height) / 2 - 40);

    sf::Text shadow = title;
    shadow.setFillColor(sf::Color(200, 0, 0));
    shadow.move(6, 6);

    sf::Text sub("Press ENTER to begin", f, 24);
    sub.setPosition(
        (win.getSize().x - sub.getGlobalBounds().width) / 2,
        title.getPosition().y + title.getGlobalBounds().height + 100);

    sf::Clock sfClock;
    bool fadeOut = false;
    float fadeTime = 1.5f;

    while (win.isOpen()) {
        sf::Event e;
        while (win.pollEvent(e)) {
            if (e.type == sf::Event::Closed) win.close();
            if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Enter) {
                fadeOut = true;
                sfClock.restart();
            }
        }

        float el = sfClock.getElapsedTime().asSeconds();
        float alpha = fadeOut ? std::max(0.f, 255.f - el / fadeTime * 255.f)
                              : std::min(255.f, el / fadeTime * 255.f);
        if (fadeOut && alpha <= 0.1f) {
            mus.stop();
            return;
        }
        if (mus.getStatus() == sf::Music::Playing && fadeOut)
            mus.setVolume(60.f * (alpha / 255.f));

        auto setA = [&](sf::Text &t, int a) {
            auto c = t.getFillColor(); c.a = a; t.setFillColor(c);
        };
        setA(title, (int)alpha);
        setA(shadow, (int)alpha);
        float pulse = std::sin(el * 3) * 0.5f + 0.5f;
        setA(sub, std::min((int)(pulse * 255), (int)alpha));

        win.clear(sf::Color(10, 10, 40));
        win.draw(shadow);
        win.draw(title);

        // Draw the pixel DNA below the title
        sf::Vector2f dnaPos(
            (win.getSize().x - 7 * 6.f) / 2.f,
            title.getPosition().y + title.getGlobalBounds().height + 40
        );
        float t = sfClock.getElapsedTime().asSeconds();
        drawPixelDNA(win, {win.getSize().x / 2.f - 24, win.getSize().y / 2.f + 50}, 6.f, t);

        win.draw(sub);
        win.display();
    }
}

//
// ---------- Main ----------
//
int main() {
    constexpr int W = 1280, H = 720, CELL = 4, FPS = 60;
    sf::RenderWindow win(sf::VideoMode(W, H), "LifeAccel — Conway's Game of Life");
    win.setFramerateLimit(FPS);

    sf::RenderWindow metrics(sf::VideoMode(320, 240), "Metrics Dashboard");
    metrics.setPosition({1320, 100});

    showTitleScreen(win);

    ThreadPool pool;
    LifeAccel life(W, H, CELL, pool);
    life.randomize(0.3);

    sf::Font font;
    font.loadFromFile("ARIAL.ttf");

    SimulationMetrics m;
    int prevLive = 0;
    sf::Clock frame, update;

    while (win.isOpen()) {
        sf::Event e;
        while (win.pollEvent(e))
            if (e.type == sf::Event::Closed) win.close();
        while (metrics.pollEvent(e))
            if (e.type == sf::Event::Closed) metrics.close();

        update.restart();
        life.updateParallel();
        m.updateMs = update.getElapsedTime().asMilliseconds();

        win.clear(sf::Color::Black);
        life.draw(win);
        win.display();

        m.frameMs = frame.restart().asMilliseconds();
        m.fps = 1000.0 / m.frameMs;
        m.avgFps = (m.avgFps * m.gen + m.fps) / (m.gen + 1);
        m.live = life.getLiveCount();
        m.delta = m.live - prevLive;
        m.gen++;
        prevLive = m.live;

        if (metrics.isOpen())
            updateMetricsWindow(metrics, m, font);
    }
    return 0;
}
